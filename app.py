"""
Contexto-style word similarity service — memory-optimized for free-tier hosting.
Uses the smaller 50-dimensional GloVe vectors and a trimmed vocabulary to stay
comfortably under low memory limits (under 200MB total).
 
Endpoints:
  POST /new-round   { "word": "optional-specific-word" }  -> picks/sets secret word, returns round_id
  POST /guess        { "round_id": "...", "guess": "dog" } -> returns rank + closeness
  GET  /reveal/<id>                                        -> returns the secret word (streamer-only peek)
  GET  /health                                              -> status check
"""
 
import os
import random
import string
import uuid
import io
import zipfile
import gc
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
 
app = Flask(__name__)
CORS(app)
 
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.50d.txt"  # smallest dimension = lowest memory use
CACHE_DIR = "/tmp/glove_cache"
CACHE_PATH = os.path.join(CACHE_DIR, GLOVE_FILE)
MAX_WORDS = 8000  # trimmed vocabulary keeps memory + CPU low while staying plenty fun
 
def download_glove():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("[Contexto] Found cached GloVe file.", flush=True)
        return
    print("[Contexto] Downloading GloVe zip (~800MB, one-time)...", flush=True)
    with urllib.request.urlopen(GLOVE_URL, timeout=300) as resp:
        with open("/tmp/glove.zip", "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    print("[Contexto] Extracting only the 50d file...", flush=True)
    with zipfile.ZipFile("/tmp/glove.zip") as z:
        z.extract(GLOVE_FILE, CACHE_DIR)
    os.remove("/tmp/glove.zip")
    print("[Contexto] GloVe ready.", flush=True)
 
print("[Contexto] Starting up...", flush=True)
download_glove()
 
print(f"[Contexto] Reading top {MAX_WORDS} words into memory...", flush=True)
words = []
vectors = []
with open(CACHE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if len(words) >= MAX_WORDS:
            break
        parts = line.rstrip().split(" ")
        word = parts[0]
        if word.isalpha() and word.islower() and 2 < len(word) < 12:
            words.append(word)
            vectors.append(np.array(parts[1:], dtype=np.float32))
 
VALID_WORDS = words
MATRIX = np.array(vectors, dtype=np.float32)
del vectors
gc.collect()
 
NORMS = np.linalg.norm(MATRIX, axis=1, keepdims=True)
NORMS[NORMS == 0] = 1e-8
MATRIX_NORM = MATRIX / NORMS
 
WORD_TO_IDX = {w: i for i, w in enumerate(VALID_WORDS)}
 
print(f"[Contexto] Ready with {len(VALID_WORDS)} words.", flush=True)
 
# In-memory store of active rounds: round_id -> { secret_word, ranks (np array) }
rounds = {}
 
def build_rankings(secret_idx):
    """Returns an array where result[i] = rank of word i relative to the secret word."""
    secret_vec = MATRIX_NORM[secret_idx]
    sims = MATRIX_NORM @ secret_vec
    order = np.argsort(-sims)
    ranks = np.empty(len(VALID_WORDS), dtype=np.int32)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks
 
@app.route("/new-round", methods=["POST"])
def new_round():
    data = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()
 
    if requested_word:
        if requested_word not in WORD_TO_IDX:
            return jsonify({"error": "Word not recognized. Try a simpler, common word."}), 400
        secret_idx = WORD_TO_IDX[requested_word]
    else:
        secret_idx = random.randrange(len(VALID_WORDS))
 
    secret_word = VALID_WORDS[secret_idx]
    round_id = str(uuid.uuid4())[:8]
    ranks = build_rankings(secret_idx)
    rounds[round_id] = {"secret_word": secret_word, "ranks": ranks}
 
    if len(rounds) > 5:
        oldest = list(rounds.keys())[0]
        del rounds[oldest]
 
    return jsonify({"round_id": round_id, "word_length": len(secret_word)})
 
@app.route("/guess", methods=["POST"])
def guess():
    data = request.get_json(force=True) or {}
    round_id = data.get("round_id")
    word = (data.get("guess") or "").strip().lower()
    word = word.translate(str.maketrans('', '', string.punctuation))
 
    if round_id not in rounds:
        return jsonify({"error": "Round not found or expired"}), 404
    if not word:
        return jsonify({"error": "Empty guess"}), 400
 
    r = rounds[round_id]
    if word == r["secret_word"]:
        return jsonify({"rank": 1, "correct": True, "word": word})
 
    if word not in WORD_TO_IDX:
        return jsonify({"error": "unknown_word", "word": word}), 200
 
    idx = WORD_TO_IDX[word]
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
