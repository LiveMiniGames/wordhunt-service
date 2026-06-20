"""
Contexto-style word similarity service.
Loads pre-trained GloVe word vectors once on startup, then serves
fast similarity rankings between a secret word and guessed words.

Endpoints:
  POST /new-round   { "word": "optional-specific-word" }  -> picks/sets secret word, returns round_id
  POST /guess        { "round_id": "...", "guess": "dog" } -> returns rank + closeness
  GET  /random-word                                        -> returns a random valid word (for "show me the word" peek)
"""

import os
import random
import string
import uuid
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim.downloader as api

app = Flask(__name__)
CORS(app)

print("[Contexto] Loading word vectors... this happens once on startup.")
model = api.load("glove-wiki-gigaword-100")
print(f"[Contexto] Loaded {len(model.key_to_index)} words.")

# Only allow common, clean alphabetic words as possible secret words / guesses
VALID_WORDS = [
    w for w in model.key_to_index.keys()
    if w.isalpha() and len(w) > 2 and len(w) < 12 and w.islower()
][:20000]  # cap to the 20k most frequent valid words for quality + speed

print(f"[Contexto] {len(VALID_WORDS)} valid words available.")

# In-memory store of active rounds: round_id -> { secret_word, rankings (word->rank dict) }
rounds = {}

def build_rankings(secret_word):
    """Pre-compute similarity rank for every valid word relative to the secret word."""
    secret_vec = model[secret_word]
    sims = []
    for w in VALID_WORDS:
        if w == secret_word:
            continue
        sim = np.dot(model[w], secret_vec) / (np.linalg.norm(model[w]) * np.linalg.norm(secret_vec))
        sims.append((w, sim))
    sims.sort(key=lambda x: -x[1])
    rank_map = {secret_word: 1}
    for i, (w, sim) in enumerate(sims):
        rank_map[w] = i + 2  # secret word is rank 1
    return rank_map

@app.route("/new-round", methods=["POST"])
def new_round():
    data = request.get_json(force=True) or {}
    requested_word = (data.get("word") or "").strip().lower()

    if requested_word:
        if requested_word not in model.key_to_index:
            return jsonify({"error": "Word not recognized. Try a simpler, common word."}), 400
        secret_word = requested_word
    else:
        secret_word = random.choice(VALID_WORDS)

    round_id = str(uuid.uuid4())[:8]
    rank_map = build_rankings(secret_word)
    rounds[round_id] = {"secret_word": secret_word, "ranks": rank_map}

    # Keep memory in check — only keep last 5 rounds
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

    if word not in model.key_to_index:
        return jsonify({"error": "unknown_word", "word": word}), 200

    rank = r["ranks"].get(word)
    if rank is None:
        # word exists in model but wasn't in our precomputed top list — compute on the fly
        secret_vec = model[r["secret_word"]]
        sim = np.dot(model[word], secret_vec) / (np.linalg.norm(model[word]) * np.linalg.norm(secret_vec))
        # Estimate rank as worse than all precomputed (rough fallback)
        rank = len(VALID_WORDS)

    return jsonify({"rank": rank, "correct": False, "word": word})

@app.route("/reveal/<round_id>", methods=["GET"])
def reveal(round_id):
    """Lets the streamer peek the secret word without ending the round."""
    if round_id not in rounds:
        return jsonify({"error": "Round not found"}), 404
    return jsonify({"secret_word": rounds[round_id]["secret_word"]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "words_loaded": len(VALID_WORDS)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
