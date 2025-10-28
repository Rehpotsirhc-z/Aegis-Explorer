import os
import sys
import json
import argparse
from typing import List, Dict, Any

from openai import OpenAI

# Load API key from gitignored local_secrets.py or ENV
try:
    from local_secrets import OPENAI_API_KEY as _OPENAI_API_KEY
except Exception:
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not _OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Missing OPENAI_API_KEY. Add Server/local_secrets.py or set env var.")
    sys.exit(1)

client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else OpenAI()

ALLOWED_CATEGORIES = {"profanity", "explicit", "drugs", "games", "gambling"}

def classify_texts_openai(texts: List[str], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """
    Returns: [{ "text": str, "flags": [ { "category": str, "confidence": float } ] }]
    """
    items = [{"id": i, "text": (t if isinstance(t, str) else str(t))[:1000]} for i, t in enumerate(texts)]

    system_prompt = (
        "You are a strict K-12 content safety classifier. "
        "For each input item, decide zero or more categories from this exact set: "
        "profanity, explicit, drugs, games, gambling. "
        "Definitions: "
        "- profanity: vulgar or obscene language and slurs; "
        "- explicit: sexual content, sexual acts, nudity, sexting; anything sexual involving minors is explicit; "
        "- drugs: illegal drugs, misuse of prescription drugs, paraphernalia; "
        "- games: references that primarily direct to or discuss web-based or online games; "
        "- gambling: betting, wagering, casinos, lotteries. "
        "If none apply, return an empty flags array. "
        "Output JSON only with a top-level object: {\"results\":[{"
        "\"id\": number, \"flags\":[{\"category\": string, \"confidence\": number}] }...]}. "
        "confidence is a number in [0,1]. Do not rewrite or summarize the text."
    )

    payload = {"items": items}

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        return [{"text": t, "flags": []} for t in texts]

    results_by_id = {r.get("id"): r for r in data.get("results", []) if isinstance(r, dict)}
    out = []
    for i, t in enumerate(texts):
        entry = results_by_id.get(i, {}) or {}
        raw_flags = entry.get("flags", []) if isinstance(entry.get("flags", []), list) else []
        flags = []
        for f in raw_flags:
            try:
                cat = str(f.get("category", "")).lower().strip()
                if cat in ALLOWED_CATEGORIES:
                    conf = float(f.get("confidence", 0.0))
                    flags.append({"category": cat, "confidence": max(0.0, min(1.0, conf))})
            except Exception:
                continue
        out.append({"text": t, "flags": flags})
    return out

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI text safety classification.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model ID to use.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON results.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", help="Path to a UTF-8 text file (one line per sample).")
    group.add_argument("--stdin", action="store_true", help="Read samples from STDIN (one per line).")
    parser.add_argument("--texts", nargs="*", help="Provide one or more text samples on the command line.")
    args = parser.parse_args()

    # Collect inputs
    texts = []
    if args.texts:
        texts = args.texts
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    elif args.stdin:
        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    else:
        # Default samples
        texts = [
            "Hello class, welcome to science!",
            "This site promotes slots, betting, and roulette bonuses.",
            "He used a lot of bad words in that post.",
            "We discussed chemistry and caffeine.",
        ]

    results = classify_texts_openai(texts, model=args.model)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Pretty print
    for r in results:
        print("\nText:", r["text"])
        if not r["flags"]:
            print("  Flags: []  (safe)")
        else:
            for f in r["flags"]:
                print(f"  - {f['category']} (confidence={f['confidence']:.2f})")

    # Summary
    counts = {}
    for r in results:
        for f in r["flags"]:
            counts[f["category"]] = counts.get(f["category"], 0) + 1
    if counts:
        print("\nSummary:", counts)
    else:
        print("\nSummary: all safe")

if __name__ == "__main__":
    main()