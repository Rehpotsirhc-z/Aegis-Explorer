import os

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
from PIL import Image
from io import BytesIO

import json
from openai import OpenAI

try:
    from local_secrets import OPENAI_API_KEY as _OPENAI_API_KEY
except Exception:
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else OpenAI()


# Helper: classify texts with OpenAI and return your expected shape
def classify_texts_openai(texts):
    """
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
        "For each input item, decide zero or more categories from this exact set: "
        "profanity, explicit, drugs, games, gambling. "
        "Definitions: "
        "- profanity: vulgar or obscene language and slurs; "
        '- explicit: sexual content, sexual acts, nudity, sexting; anything sexual involving minors is explicit; any language hinting at NSFW (e.g. "Bet her cheeks are showing", or "Sore asshohohole"); ALSO INCLUDES: graphic violence, gore, severe physical harm, detailed depictions of injury or death (e.g. "mixed with skin and blood"); ALSO INCLUDES: content that encourages, directs, or nudges the viewer toward NSFW material elsewhere (e.g., comments linking to NSFW subreddits or websites ("This is literally r/meatcrayon (NSFW sub)"), suggestions to "check the uncensored version," or remarks implying explicit content off-platform such as "this gets way worse later" or "look up the uncut scene").'
        '- drugs: illegal drugs, misuse of prescription drugs, paraphernalia; any mentions of drug use, buying, selling, or seeking drugs (e.g. "cocaine" or "marijuana")'
        "- games: references that primarily direct to or discuss games; promoting games or purchases in games; "
        "- gambling: betting, wagering, casinos, lotteries. (e.g. 'bet', 'wager', 'casino', 'poker', 'lottery', 'slots', 'BetMGM'). "
        "If none apply, return an empty flags array. "
        'Output JSON only with a top-level object: {"results":[{'
        '"id": number, "flags":[{"category": string, "confidence": number}] }...]}. '
        "confidence is a number in [0,1]. Do not rewrite or summarize the text."
    )

    user_payload = {"items": items}

    # Use JSON mode for structured output
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
        # Fallback: everything safe
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


# set number of threads to 8
# torch.set_num_threads(8)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
image_model_path = Path("models/image/model_v9.pt")
img_model = YOLO(image_model_path)
img_model.to(device)


def saveImgToFile(img, path):
    img.save(path)


@app.post("/predict_image")
async def predict_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes))

        results = img_model(img)
        predictions = results[0].boxes

        response = {
            "predictions": [
                {
                    "class": img_model.names[int(pred.cls)],
                    "confidence": float(pred.conf),
                }
                for pred in predictions
            ]
        }

        return JSONResponse(response)

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
        results = classify_texts_openai(texts)
        return JSONResponse(results)
    except Exception as e:
        print("OpenAI classification error:", e)
        return JSONResponse([{"text": t, "flags": []} for t in texts])
