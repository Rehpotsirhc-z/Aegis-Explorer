import os
import json
import hashlib
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache for text classifications (avoids re-calling OpenAI for same text)
text_cache = OrderedDict()
TEXT_CACHE_MAX = 5000



def cache_get(text):
    key = hashlib.md5(text.encode()).hexdigest()
    if key in text_cache:
        text_cache.move_to_end(key)
        return text_cache[key]
    return None


def cache_set(text, result):
    key = hashlib.md5(text.encode()).hexdigest()
    if len(text_cache) >= TEXT_CACHE_MAX:
        text_cache.popitem(last=False)
    text_cache[key] = result


def classify_texts_openai(texts):
    """
    Classify texts using OpenAI.
    Returns: list of { "text": str, "flags": [ { "category": str, "confidence": float } ] }
    Categories allowed: profanity, explicit, drugs, games, gambling
    """
    # Hard clamp input length for cost/safety
    items = [
        {"id": i, "text": (t if isinstance(t, str) else str(t))[:2000]}
        for i, t in enumerate(texts)
    ]

    system_prompt = (
        "You are a strict K-12 content safety classifier. "
        "Classify each item into zero or more categories: profanity, explicit, drugs, games, gambling. "
        "- profanity: vulgar/obscene language and slurs. "
        "- explicit: sexual content, nudity, sexting; anything sexual involving minors; "
        "graphic violence, gore, detailed injury/death; "
        'directing users to NSFW material (e.g. "check the uncensored version", linking NSFW subreddits). '
        "- drugs: illegal drugs, drug paraphernalia, promoting/glorifying drug use or selling. "
        "Ignore educational/news/health contexts (e.g. 'drug prevention program'). "
        "- games: promoting or directing to video/online/mobile games or gaming platforms. "
        "Ignore casual use (e.g. 'game plan', 'ball game'). "
        "- gambling: betting, wagering, casinos, lotteries, sports betting, slots. "
        "Ignore idioms (e.g. 'I bet you're right', 'you bet'). "
        "Empty flags array if none apply. "
        'Output JSON: {"results":[{"id":number,"flags":[{"category":string,"confidence":number}]}...]}. '
        "confidence in [0,1]. Do not rewrite the text."
    )

    user_payload = {"items": items}

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        return [{"text": t, "flags": []} for t in texts]

    results_by_id = {
        r.get("id"): r for r in data.get("results", []) if isinstance(r, dict)
    }
    allowed = {"profanity", "explicit", "drugs", "games", "gambling"}

    out = []
    for i, t in enumerate(texts):
        entry = results_by_id.get(i, {})
        flags_raw = (
            entry.get("flags", []) if isinstance(entry.get("flags", []), list) else []
        )
        flags = []
        for f in flags_raw:
            try:
                cat = str(f.get("category", "")).lower().strip()
                if cat in allowed:
                    conf = float(f.get("confidence", 0.0))
                    conf = max(0.0, min(1.0, conf))
                    flags.append({"category": cat, "confidence": conf})
            except Exception:
                continue
        out.append({"text": t, "flags": flags})
    return out


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- YOLO Image Classification ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_model_path = Path("AI_Code/model/model_v9.pt")
img_model = YOLO(image_model_path)
img_model.to(device)


def run_yolo(url):
    """Run YOLO inference in a thread (synchronous, blocks event loop otherwise)."""
    import base64
    import requests
    from PIL import Image
    from io import BytesIO

    if url.startswith("data:"):
        # data:[<mediatype>][;base64],<data>
        try:
            _, encoded = url.split(",", 1)
            img = Image.open(BytesIO(base64.b64decode(encoded)))
        except Exception as e:
            raise ValueError(f"Invalid data URI: {e}")
    else:
        # Download image and pass as PIL Image to avoid YOLO treating URLs as video streams
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))

    results = img_model(img, save=False, verbose=False)
    predictions = results[0].boxes
    return [
        {
            "class": img_model.names[int(pred.cls)],
            "confidence": float(pred.conf),
        }
        for pred in predictions
    ]


@app.post("/predict_image")
async def predict_image(payload: dict):
    try:
        url = payload.get("url")
        if not url:
            return JSONResponse({"error": "Missing 'url' field"}, status_code=400)

        import asyncio
        loop = asyncio.get_event_loop()
        preds = await loop.run_in_executor(None, run_yolo, url)

        return JSONResponse({"predictions": preds})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict_text")
async def predict_text(payload: dict):
    if "texts" not in payload:
        return JSONResponse({"error": "Missing 'texts' field"}, status_code=400)

    texts = payload["texts"]
    if not isinstance(texts, list):
        return JSONResponse({"error": "'texts' must be a list"}, status_code=400)

    try:
        # Check cache first - only send uncached texts to OpenAI
        results = [None] * len(texts)
        uncached = []
        uncached_indices = []

        for i, t in enumerate(texts):
            cached = cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append(t)
                uncached_indices.append(i)

        if uncached:
            fresh = classify_texts_openai(uncached)
            for idx, result in zip(uncached_indices, fresh):
                results[idx] = result
                cache_set(result["text"], result)

        return JSONResponse(results)
    except Exception as e:
        print("OpenAI classification error:", e)
        return JSONResponse([{"text": t, "flags": []} for t in texts])
