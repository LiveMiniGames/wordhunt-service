"""
Contexto-style word similarity service.
Uses GloVe 6B 300d vectors — full vocabulary (~400K words), 300 dimensions.
Much more accurate semantic similarity than the previous 50d/8000-word version.

Endpoints:
  POST /new-round   { "word": "optional-specific-word" }  -> picks/sets secret word, returns round_id
  POST /guess        { "round_id": "...", "guess": "dog" } -> returns rank + closeness
  GET  /reveal/<id>                                        -> returns the secret word (streamer-only peek)
  GET  /health                                             -> status check
"""

import os
import random
import string
import uuid
import zipfile
import gc
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# GloVe 6B 300d — best quality that still fits in ~512MB free tier RAM
GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.300d.txt"
CACHE_DIR  = "/tmp/glove_cache"
CACHE_PATH = os.path.join(CACHE_DIR, GLOVE_FILE)

# Word filters — keep common single words, exclude super-short/long/junk
MIN_LEN = 3
MAX_LEN = 14

def download_glove():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("[Contexto] Found cached GloVe 300d file.", flush=True)
        return
    zip_path = "/tmp/glove.zip"
    print("[Contexto] Downloading GloVe 6B zip (~860MB, one-time)...", flush=True)
    with urllib.request.urlopen(GLOVE_URL, timeout=600) as resp:
        with open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    print("[Contexto] Extracting 300d file...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(GLOVE_FILE, CACHE_DIR)
    os.remove(zip_path)
    print("[Contexto] GloVe 300d ready.", flush=True)

print("[Contexto] Starting up...", flush=True)
download_glove()

print("[Contexto] Loading vectors into memory...", flush=True)
words   = []
vectors = []

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word  = parts[0]
        # Keep only clean alphabetic words of reasonable length
        if word.isalpha() and word.islower() and MIN_LEN <= len(word) <= MAX_LEN:
            words.append(word)
            vectors.append(np.array(parts[1:], dtype=np.float32))

VALID_WORDS = words
MATRIX      = np.array(vectors, dtype=np.float32)
del vectors
gc.collect()

# Pre-normalise all vectors so similarity = just a dot product (fast)
NORMS            = np.linalg.norm(MATRIX, axis=1, keepdims=True)
NORMS[NORMS == 0] = 1e-8
MATRIX_NORM      = MATRIX / NORMS
del MATRIX
gc.collect()

WORD_TO_IDX = {w: i for i, w in enumerate(VALID_WORDS)}

print(f"[Contexto] Ready — {len(VALID_WORDS):,} words loaded with 300d vectors.", flush=True)

# ─── Round storage ────────────────────────────────────────────────────────────
rounds = {}     # round_id -> { secret_word, ranks }

def build_rankings(secret_idx):
    """
    Computes cosine similarity between the secret word and every word in the
    vocabulary, then returns an array where result[i] = rank of word i
    (1 = identical / exact answer, higher = less similar).
    """
    secret_vec = MATRIX_NORM[secret_idx]
    sims       = MATRIX_NORM @ secret_vec          # dot product of normalised = cosine sim
    order      = np.argsort(-sims)                  # descending similarity
    ranks      = np.empty(len(VALID_WORDS), dtype=np.int32)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/new-round", methods=["POST"])
def new_round():
    data           = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()

    if requested_word:
        if requested_word not in WORD_TO_IDX:
            return jsonify({"error": "Word not in vocabulary. Try a different common word."}), 400
        secret_idx = WORD_TO_IDX[requested_word]
    else:
        # Only pick from words that make good game targets:
        # concrete nouns that chat can meaningfully guess around.
        # We avoid verbs, adjectives, and abstract words by using a curated
        # list of categories. Words in GloVe are sorted by frequency so the
        # top ~30K are the most common — but we further filter by a
        # hand-picked suffix/pattern blocklist to remove obvious non-nouns.

        VERB_SUFFIXES    = ('ing','ed','tion','sion','ment','ness','ity','ism',
                             'ize','ise','ify','ate','fy','ous','ful','less',
                             'able','ible','ive','ary','ery','ory','ward','wise',
                             'ly','est','er')
        BLOCKED_WORDS    = {
            # super-abstract concepts that score badly in word games
            'the','and','for','are','but','not','you','all','can','her','was',
            'one','our','out','day','get','has','him','his','how','man','new',
            'now','old','see','two','way','who','boy','did','its','let','put',
            'say','she','too','use','had','may','been','come','does','from',
            'into','just','like','long','make','many','more','most','much',
            'need','only','over','same','some','take','than','that','them',
            'then','they','this','time','very','well','what','when','will',
            'with','also','back','been','call','came','each','even','find',
            'give','good','hand','here','high','keep','know','last','left',
            'life','live','look','made','mind','move','must','name','near',
            'next','only','open','part','play','real','said','show','side',
            'such','sure','tell','than','them','till','told','turn','upon',
            'used','want','went','were','whom','year','your',
            # past tenses / gerunds that slipped through suffix check
            'promised','happened','decided','started','wanted','needed',
            'seemed','looked','asked','called','tried','moved','turned',
            'helped','lived','loved','worked','played','stopped','stayed',
        }

        candidates = []
        # GloVe 6B is sorted roughly by frequency — top 30K words are common
        pool = min(30000, len(VALID_WORDS))
        for i in range(pool):
            w = VALID_WORDS[i]
            # skip blocked
            if w in BLOCKED_WORDS:
                continue
            # skip anything ending in a verb/adjective suffix
            if any(w.endswith(s) for s in VERB_SUFFIXES):
                continue
            # keep words 4-10 chars — sweet spot for game words
            if not (4 <= len(w) <= 10):
                continue
            candidates.append(i)

        secret_idx = random.choice(candidates) if candidates else random.randrange(min(30000, len(VALID_WORDS)))

    secret_word = VALID_WORDS[secret_idx]
    round_id    = str(uuid.uuid4())[:8]
    ranks       = build_rankings(secret_idx)

    rounds[round_id] = {"secret_word": secret_word, "ranks": ranks}

    # Keep memory bounded — evict oldest round if we accumulate too many
    if len(rounds) > 10:
        oldest = next(iter(rounds))
        del rounds[oldest]

    return jsonify({"round_id": round_id, "word_length": len(secret_word)})


@app.route("/guess", methods=["POST"])
def guess():
    data     = request.get_json(force=True) or {}
    round_id = data.get("round_id")
    word     = (data.get("guess") or "").strip().lower()
    word     = word.translate(str.maketrans('', '', string.punctuation))

    if round_id not in rounds:
        return jsonify({"error": "Round not found or expired"}), 404
    if not word:
        return jsonify({"error": "Empty guess"}), 400

    r = rounds[round_id]

    if word == r["secret_word"]:
        return jsonify({"rank": 1, "correct": True, "word": word})

    if word not in WORD_TO_IDX:
        return jsonify({"error": "unknown_word", "word": word}), 200

    idx  = WORD_TO_IDX[word]
    rank = int(r["ranks"][idx])
    return jsonify({"rank": rank, "correct": False, "word": word})


@app.route("/reveal/<round_id>", methods=["GET"])
def reveal(round_id):
    if round_id not in rounds:
        return jsonify({"error": "Round not found"}), 404
    return jsonify({"secret_word": rounds[round_id]["secret_word"]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "words_loaded": len(VALID_WORDS)})


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Word Hunt service running", "words_loaded": len(VALID_WORDS)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
