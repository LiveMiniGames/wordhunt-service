"""
Contexto-style word similarity service.
Loads pre-trained GloVe word vectors directly (no gensim needed), then serves
fast similarity rankings between a secret word and guessed words.
 
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
import urllib.request
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
 
app = Flask(__name__)
CORS(app)
 
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
CACHE_DIR = "/tmp/glove_cache"
CACHE_PATH = os.path.join(CACHE_DIR, GLOVE_FILE)
 
def download_glove():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
        print("[Contexto] Found cached GloVe file.")
        return
    print("[Contexto] Downloading GloVe vectors (this happens once)...")
    with urllib.request.urlopen(GLOVE_URL, timeout=120) as resp:
        data = resp.read()
    print("[Contexto] Extracting...")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract(GLOVE_FILE, CACHE_DIR)
    print("[Contexto] GloVe ready.")
 
print("[Contexto] Loading word vectors... this happens once on startup.")
download_glove()
 
word_to_vec = {}
with open(CACHE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        if word.isalpha() and word.islower() and 2 < len(word) < 12:
            vec = np.array(parts[1:], dtype=np.float32)
            word_to_vec[word] = vec
 
print(f"[Contexto] Loaded {len(word_to_vec)} usable words.")
 
# Limit to the most common ~20k words for quality + speed (GloVe file is frequency-sorted)
VALID_WORDS = list(word_to_vec.keys())[:20000]
print(f"[Contexto] {len(VALID_WORDS)} valid words available.")
 
# Pre-normalize vectors for fast cosine similarity via dot product
VALID_MATRIX = np.array([word_to_vec[w] for w in VALID_WORDS], dtype=np.float32)
VALID_NORMS = np.linalg.norm(VALID_MATRIX, axis=1, keepdims=True)
VALID_MATRIX_NORMALIZED = VALID_MATRIX / VALID_NORMS
 
WORD_INDEX = {w: i for i, w in enumerate(VALID_WORDS)}
 
# In-memory store of active rounds: round_id -> { secret_word, rank_lookup }
rounds = {}
 
def build_rankings(secret_word):
    """Compute similarity rank for every valid word relative to the secret word."""
    secret_vec = word_to_vec[secret_word]
    secret_norm = secret_vec / np.linalg.norm(secret_vec)
    sims = VALID_MATRIX_NORMALIZED @ secret_norm  # cosine similarity to all words at once
    order = np.argsort(-sims)  # descending order, most similar first
 
    rank_lookup = {}
    rank = 2  # secret word itself is rank 1
    for idx in order:
        w = VALID_WORDS[idx]
        if w == secret_word:
            continue
        rank_lookup[w] = rank
        rank += 1
    rank_lookup[secret_word] = 1
    return rank_lookup
 
@app.route("/new-round", methods=["POST"])
def new_round():
    data = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()
 
    if requested_word:
        if requested_word not in word_to_vec:
            return jsonify({"error": "Word not recognized. Try a simpler, common word."}), 400
        secret_word = requested_word
    else:
        secret_word = random.choice(VALID_WORDS)
 
    round_id = str(uuid.uuid4())[:8]
    rank_lookup = build_rankings(secret_word)
    rounds[round_id] = {"secret_word": secret_word, "ranks": rank_lookup}
 
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
 
    if word not in word_to_vec:
        return jsonify({"error": "unknown_word", "word": word}), 200
 
    rank = r["ranks"].get(word, len(VALID_WORDS) + 1)
    return jsonify({"rank": rank, "correct": False, "word": word})
 
@app.route("/reveal/<round_id>", methods=["GET"])
def reveal(round_id):
    if round_id not in rounds:
        return jsonify({"error": "Round not found"}), 404
    return jsonify({"secret_word": rounds[round_id]["secret_word"]})
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "words_loaded": len(VALID_WORDS)})
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
