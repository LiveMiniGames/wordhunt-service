"""
Contexto-style word similarity service.
Uses GloVe 6B 50d vectors — full vocabulary (~180K filtered words).
50d keeps memory well under Railway free tier (512MB).
Much better than the old 8000-word trimmed version because we now use
the complete vocabulary, so guesses are rarely "not recognized" and
rankings are far more accurate across the full word space.
"""

import os, random, string, uuid, zipfile, gc, urllib.request
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"   # 50d = small RAM footprint, full vocab
CACHE_DIR  = "/tmp/glove_cache"
CACHE_PATH = os.path.join(CACHE_DIR, GLOVE_FILE)

def download_glove():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("[Contexto] Found cached GloVe file.", flush=True)
        return
    zip_path = "/tmp/glove.zip"
    print("[Contexto] Downloading GloVe 6B zip (~820MB, one-time)...", flush=True)
    with urllib.request.urlopen(GLOVE_URL, timeout=600) as resp:
        with open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk: break
                out.write(chunk)
    print("[Contexto] Extracting 50d file...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(GLOVE_FILE, CACHE_DIR)
    os.remove(zip_path)
    print("[Contexto] GloVe ready.", flush=True)

print("[Contexto] Starting up...", flush=True)
download_glove()

print("[Contexto] Loading vectors...", flush=True)
words, vectors = [], []
with open(CACHE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word  = parts[0]
        if word.isalpha() and word.islower() and 3 <= len(word) <= 14:
            words.append(word)
            vectors.append(np.array(parts[1:], dtype=np.float32))

VALID_WORDS = words
MATRIX      = np.array(vectors, dtype=np.float32)
del vectors; gc.collect()

NORMS             = np.linalg.norm(MATRIX, axis=1, keepdims=True)
NORMS[NORMS == 0] = 1e-8
MATRIX_NORM       = MATRIX / NORMS
del MATRIX; gc.collect()

WORD_TO_IDX = {w: i for i, w in enumerate(VALID_WORDS)}

print(f"[Contexto] Ready — {len(VALID_WORDS):,} words.", flush=True)

# ── Word pool for random game picks ──────────────────────────────────────────
# Suffix patterns that indicate non-nouns (verbs, adjectives, adverbs, etc.)
BAD_SUFFIXES = (
    'ing','tion','sion','ment','ness','ity','ism','ize','ise','ify',
    'ate','ous','ful','less','able','ible','ive','ary','ery','ory',
    'ward','wise','ly','est','ied','ied',
)
# Common past tense -ed words and abstract words that slipped through
BLOCKLIST = {
    'promised','happened','decided','started','wanted','needed','seemed',
    'looked','asked','called','tried','moved','turned','helped','lived',
    'loved','worked','played','stopped','stayed','passed','pulled','placed',
    'loved','used','named','based','faced','liked','changed','believed',
    'opened','closed','showed','given','taken','known','gone','said',
    'came','went','been','were','have','also','just','more','much',
    'some','very','well','than','then','when','only','back','even',
    'also','each','most','such','this','that','they','them','from',
    'with','into','upon','over','your','our','his','her','its','the',
    'and','for','but','not','you','all','can','was','one','out','day',
    'get','has','him','how','man','new','now','old','see','two','way',
    'who','boy','did','let','put','say','she','too','had','may',
}

GAME_POOL = [
    i for i, w in enumerate(VALID_WORDS)
    if i < 40000                                  # stay in common-word range
    and 4 <= len(w) <= 10                         # sweet spot length
    and w not in BLOCKLIST
    and not any(w.endswith(s) for s in BAD_SUFFIXES)
]
print(f"[Contexto] Game word pool: {len(GAME_POOL):,} words.", flush=True)

# ── Round storage ─────────────────────────────────────────────────────────────
rounds = {}

def build_rankings(secret_idx):
    secret_vec = MATRIX_NORM[secret_idx]
    sims       = MATRIX_NORM @ secret_vec
    order      = np.argsort(-sims)
    ranks      = np.empty(len(VALID_WORDS), dtype=np.int32)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/new-round", methods=["POST"])
def new_round():
    data           = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()

    if requested_word:
        if requested_word not in WORD_TO_IDX:
            return jsonify({"error": "Word not in vocabulary. Try a different common word."}), 400
        secret_idx = WORD_TO_IDX[requested_word]
    else:
        secret_idx = random.choice(GAME_POOL)

    secret_word          = VALID_WORDS[secret_idx]
    round_id             = str(uuid.uuid4())[:8]
    rounds[round_id]     = {"secret_word": secret_word, "ranks": build_rankings(secret_idx)}

    if len(rounds) > 10:
        del rounds[next(iter(rounds))]

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

    return jsonify({"rank": int(r["ranks"][WORD_TO_IDX[word]]), "correct": False, "word": word})


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
    return jsonify({"status": "Contexto service running", "words_loaded": len(VALID_WORDS)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
