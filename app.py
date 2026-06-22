"""
Contexto-style word similarity service.
GloVe 6B 50d, full filtered vocabulary.
Startup downloads in background so Railway health check passes immediately.
"""

import os, random, string, uuid, zipfile, gc, urllib.request, threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"
CACHE_DIR  = "/tmp/glove_cache"
CACHE_PATH = os.path.join(CACHE_DIR, GLOVE_FILE)

# ── Global state ──────────────────────────────────────────────────────────────
VALID_WORDS  = []
MATRIX_NORM  = None
WORD_TO_IDX  = {}
GAME_POOL    = []
ready        = False   # True once vectors are loaded
rounds       = {}

BAD_SUFFIXES = (
    'ing','tion','sion','ment','ness','ity','ism','ize','ise','ify',
    'ate','ous','ful','less','able','ible','ive','ary','ery','ory',
    'ward','wise','ly','est',
)
BLOCKLIST = {
    'promised','happened','decided','started','wanted','needed','seemed',
    'looked','asked','called','tried','moved','turned','helped','lived',
    'loved','worked','played','stopped','stayed','passed','pulled','placed',
    'used','named','based','faced','liked','changed','believed','opened',
    'closed','showed','given','taken','known','said','came','went','been',
    'were','have','also','just','more','much','some','very','well','than',
    'then','when','only','back','even','each','most','such','this','that',
    'they','them','from','with','into','upon','over','your','our','his',
    'her','its','the','and','for','but','not','you','all','can','was',
    'one','out','day','get','has','him','how','man','new','now','old',
    'see','two','way','who','boy','did','let','put','say','she','too',
    'had','may',
}

# ── Download + load in a background thread ────────────────────────────────────
def load_vectors():
    global VALID_WORDS, MATRIX_NORM, WORD_TO_IDX, GAME_POOL, ready

    # 1. Download if needed
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(CACHE_PATH):
        print("[Contexto] Downloading GloVe zip (one-time)...", flush=True)
        zip_path = "/tmp/glove.zip"
        try:
            with urllib.request.urlopen(GLOVE_URL, timeout=600) as resp:
                with open(zip_path, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk: break
                        out.write(chunk)
            with zipfile.ZipFile(zip_path) as z:
                z.extract(GLOVE_FILE, CACHE_DIR)
            os.remove(zip_path)
            print("[Contexto] Download complete.", flush=True)
        except Exception as e:
            print(f"[Contexto] Download failed: {e}", flush=True)
            return
    else:
        print("[Contexto] Using cached GloVe file.", flush=True)

    # 2. Load vectors
    print("[Contexto] Loading vectors...", flush=True)
    words, vectors = [], []
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if word.isalpha() and word.islower() and 3 <= len(word) <= 14:
                    words.append(word)
                    vectors.append(np.array(parts[1:], dtype=np.float32))
    except Exception as e:
        print(f"[Contexto] Load failed: {e}", flush=True)
        return

    matrix = np.array(vectors, dtype=np.float32)
    del vectors; gc.collect()

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_matrix = matrix / norms
    del matrix; gc.collect()

    word_to_idx = {w: i for i, w in enumerate(words)}

    game_pool = [
        i for i, w in enumerate(words)
        if i < 40000
        and 4 <= len(w) <= 10
        and w not in BLOCKLIST
        and not any(w.endswith(s) for s in BAD_SUFFIXES)
    ]

    # Assign to globals atomically
    VALID_WORDS  = words
    MATRIX_NORM  = norm_matrix
    WORD_TO_IDX  = word_to_idx
    GAME_POOL    = game_pool
    ready        = True
    print(f"[Contexto] Ready — {len(VALID_WORDS):,} words, {len(GAME_POOL):,} game words.", flush=True)

# Start loading immediately in background
threading.Thread(target=load_vectors, daemon=True).start()

# ── Helpers ───────────────────────────────────────────────────────────────────
def build_rankings(secret_idx):
    secret_vec = MATRIX_NORM[secret_idx]
    sims       = MATRIX_NORM @ secret_vec
    order      = np.argsort(-sims)
    ranks      = np.empty(len(VALID_WORDS), dtype=np.int32)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    # Always returns 200 so Railway considers the service up
    return jsonify({"status": "ok" if ready else "loading", "ready": ready,
                    "words_loaded": len(VALID_WORDS)})

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Contexto service", "ready": ready,
                    "words_loaded": len(VALID_WORDS)})

@app.route("/new-round", methods=["POST"])
def new_round():
    if not ready:
        return jsonify({"error": "Service is still loading word vectors — try again in a minute."}), 503

    data           = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()

    if requested_word:
        if requested_word not in WORD_TO_IDX:
            return jsonify({"error": "Word not in vocabulary. Try a different common word."}), 400
        secret_idx = WORD_TO_IDX[requested_word]
    else:
        secret_idx = random.choice(GAME_POOL)

    secret_word      = VALID_WORDS[secret_idx]
    round_id         = str(uuid.uuid4())[:8]
    rounds[round_id] = {"secret_word": secret_word, "ranks": build_rankings(secret_idx)}

    if len(rounds) > 10:
        del rounds[next(iter(rounds))]

    return jsonify({"round_id": round_id, "word_length": len(secret_word)})

@app.route("/guess", methods=["POST"])
def guess():
    if not ready:
        return jsonify({"error": "Service is still loading"}), 503

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
