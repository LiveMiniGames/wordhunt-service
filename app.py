"""
Contexto-style word similarity service.
Loads ONLY the curated ~500 game words + a ~20K common word guess vocabulary.
Tiny memory footprint, no crashes, fast ranking.
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

# Global state
VALID_WORDS = []   # all loaded words (game words + common guess words)
MATRIX_NORM = None # normalised vectors for all loaded words
WORD_TO_IDX = {}   # word -> index in MATRIX_NORM
GAME_POOL   = []   # indices of curated game words
ready       = False
rounds      = {}

# ── Curated game words — concrete nouns only ──────────────────────────────────
CURATED_WORDS = {
    'amber','ankle','ant','ape','apple','arm','ash','avocado','backpack','bacon',
    'bag','ball','banana','bark','barn','baseball','basil','basket','bass','bat',
    'bath','bathroom','battery','beach','bear','beaver','bed','bedroom','bee','beef',
    'beetle','belt','bicycle','bike','birch','bird','blizzard','blueberry','board','boat',
    'bolt','bone','book','boots','bowl','boxing','bracelet','brain','branch','bread',
    'bridge','broccoli','broom','brush','bubble','bucket','burger','bus','bush','butter',
    'butterfly','button','cabbage','cable','cake','calf','camel','canyon','cap','car',
    'caramel','card','carrot','castle','cat','cave','cedar','ceiling','celery','chain',
    'chair','charger','cheek','cheese','cherry','chest','chicken','chili','chin',
    'chocolate','church','cliff','clock','cloud','clover','coat','coconut','coffee',
    'comb','cookie','coral','corn','cotton','court','cow','crab','cream','crow',
    'crystal','cucumber','cup','daisy','deer','desert','desk','dew','diamond','dice',
    'dock','dog','dolphin','donut','door','dove','dress','drill','drum','duck',
    'dust','eagle','ear','earring','egg','elbow','elephant','elm','ember','eye',
    'face','factory','farm','fern','field','finger','fish','flame','flamingo',
    'flood','floor','flour','flower','flute','foam','fog','foot','football','forest',
    'fork','fox','fridge','frog','frost','fruit','garage','garden','garlic','ginger',
    'giraffe','glacier','glass','glasses','gloves','glue','goal','goat','golf','goose',
    'gorilla','grape','grass','guitar','hail','hair','ham','hammer','hamster','hand',
    'harp','hat','hawk','head','heart','heel','hen','hip','hippo','hockey','honey',
    'hoodie','hoop','horse','hospital','house','hurricane','ice','island','ivory',
    'jacket','jaw','jeans','jellyfish','juice','jungle','kangaroo','ketchup','keyboard',
    'kidney','king','kitchen','kitten','knee','knife','koala','ladder','lake','lamb',
    'lamp','leaf','leg','lemon','lettuce','lightning','lily','lime','lion','lip',
    'liver','lobster','lung','mango','maple','meadow','melon','milk','mint','mirror',
    'mist','monkey','moon','mop','motorcycle','mountain','mouse','mouth','muffin',
    'muscle','mushroom','mustard','nail','neck','necklace','net','noodle','nose',
    'notebook','nut','oak','ocean','octopus','onion','orange','orchid','otter','oven',
    'owl','paint','pan','pancake','panda','pants','paper','park','parrot','pasta',
    'path','peach','pear','pearl','pelican','pen','pencil','penguin','pepper','petal',
    'phone','piano','pie','pig','pigeon','pine','pineapple','pizza','plane','planet',
    'plate','plug','plum','pond','pool','poppy','pork','pot','potato','puppy',
    'purse','queen','rabbit','raccoon','racket','rain','rainbow','raspberry','rat',
    'razor','rhino','rice','ring','river','road','robin','rock','roof','rooster',
    'root','rope','rose','ruby','ruler','salad','salmon','salt','sand','sandals',
    'sandwich','sauce','sausage','saw','scarf','school','scissors','screen','screw',
    'sea','seal','seed','shadow','shampoo','shark','sheep','shelf','ship','shirt',
    'shoes','shorts','shoulder','shower','shrimp','silk','sink','skin','skirt',
    'skunk','smoke','snail','snake','sneakers','snow','soap','soccer','socks','sofa',
    'soil','soup','spark','sparrow','spider','spinach','spoon','squid','squirrel',
    'stairs','star','starfish','stew','stomach','stone','storm','stove','strawberry',
    'stream','street','sugar','suit','sun','sunflower','sunrise','swamp','sweater',
    'switch','table','tape','tennis','thorn','thumb','thunder','tiger','toe','toilet',
    'tomato','tongue','tooth','toothbrush','tornado','towel','tower','town','train',
    'tree','trout','truck','trumpet','trunk','tulip','tuna','tunnel','turkey','turtle',
    'valley','vanilla','velvet','vest','village','vine','violet','violin','volcano',
    'waffle','wall','wallet','wasp','watch','water','wave','whale','wheat','willow',
    'wind','window','wire','wolf','wool','worm','wrench','wrist','yogurt','zebra',
}

# Common English words people might guess — loaded for ranking purposes
# We cap at 15K to stay well under memory limits
MAX_VOCAB = 15000

def load_vectors():
    global VALID_WORDS, MATRIX_NORM, WORD_TO_IDX, GAME_POOL, ready

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

    print("[Contexto] Loading vectors...", flush=True)
    words, vectors = [], []
    seen = 0
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if not (word.isalpha() and word.islower() and 2 <= len(word) <= 14):
                    continue
                # Always include curated words; also include top common words up to MAX_VOCAB
                if word in CURATED_WORDS or seen < MAX_VOCAB:
                    words.append(word)
                    vectors.append(np.array(parts[1:], dtype=np.float32))
                    if word not in CURATED_WORDS:
                        seen += 1
                if seen >= MAX_VOCAB and all(w in {ww for ww in words} for w in CURATED_WORDS):
                    break  # got everything we need
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
    game_pool   = [word_to_idx[w] for w in CURATED_WORDS if w in word_to_idx]

    VALID_WORDS = words
    MATRIX_NORM = norm_matrix
    WORD_TO_IDX = word_to_idx
    GAME_POOL   = game_pool
    ready       = True
    print(f"[Contexto] Ready — {len(VALID_WORDS):,} total words, {len(GAME_POOL):,} game words.", flush=True)

threading.Thread(target=load_vectors, daemon=True).start()

def get_rank(secret_vec, guess_idx):
    # Matrix is now tiny (~15K words) so this is fast and safe
    guess_sim = float(MATRIX_NORM[guess_idx] @ secret_vec)
    all_sims  = MATRIX_NORM @ secret_vec
    return int(np.sum(all_sims > guess_sim)) + 1

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok" if ready else "loading",
                    "ready": ready, "words_loaded": len(VALID_WORDS)})

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "Contexto service", "ready": ready,
                    "words_loaded": len(VALID_WORDS)})

@app.route("/new-round", methods=["POST"])
def new_round():
    if not ready:
        return jsonify({"error": "Service is still loading — try again in a minute."}), 503
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
    rounds[round_id] = {"secret_word": secret_word, "secret_vec": MATRIX_NORM[secret_idx].copy()}
    if len(rounds) > 20:
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
    rank = get_rank(r["secret_vec"], WORD_TO_IDX[word])
    return jsonify({"rank": rank, "correct": False, "word": word})

@app.route("/reveal/<round_id>", methods=["GET"])
def reveal(round_id):
    if round_id not in rounds:
        return jsonify({"error": "Round not found"}), 404
    return jsonify({"secret_word": rounds[round_id]["secret_word"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
