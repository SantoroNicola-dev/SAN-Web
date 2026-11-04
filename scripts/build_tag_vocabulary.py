"""
Builds the curated multi-label English tag vocabulary (~4,600+ tags) for SAN-Web.
Combines the Recognize Anything (RAM) 4,585 open-vocabulary tag set with SAN-1's
specialized categories, deduplicating and producing:
  - models/labels.json (structured metadata with index, tag, id, group)
  - models/labels.txt  (plain 1-line-per-tag list for Unity TextAsset / lightweight web loading)
"""

import json
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SAN1_CATEGORIES_PATH = Path(r"D:\Projects\VibeProject\SAN-1\config\categories.json")

RAM_TAG_LIST_URLS = [
    "https://raw.githubusercontent.com/xinyu1205/recognize-anything/main/ram/data/ram_tag_list.txt",
    "https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/raw/main/ram/data/ram_tag_list.txt",
]


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def fetch_ram_tags() -> list[str]:
    cache_file = MODELS_DIR / "raw_ram_tags.txt"
    if cache_file.exists():
        lines = [l.strip() for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) >= 4000:
            return lines

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for url in RAM_TAG_LIST_URLS:
        try:
            print(f"[vocab] Downloading RAM 4,585 tag list from {url} ...")
            req = urllib.request.urlopen(url, timeout=15)
            content = req.read().decode("utf-8")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if len(lines) >= 4000:
                cache_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                return lines
        except Exception as exc:
            print(f"[vocab] Warning: failed to fetch {url}: {exc}")

    raise RuntimeError("Could not download RAM tag list.")


def load_san1_tags() -> list[dict]:
    if not SAN1_CATEGORIES_PATH.exists():
        return []
    data = json.loads(SAN1_CATEGORIES_PATH.read_text(encoding="utf-8"))
    extra = []
    for cat in data.get("categories", []):
        label = cat.get("label", "").strip()
        if label:
            extra.append({
                "label": label,
                "group": cat.get("group", "general"),
                "description": cat.get("description", ""),
                "prompts": cat.get("prompts", []),
            })
    return extra


def build_vocabulary() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ram_tags = fetch_ram_tags()
    san1_cats = load_san1_tags()

    seen_lower = set()
    entries = []

    # 1. Add all 4,585 Recognize Anything (RAM) English tags first
    for tag in ram_tags:
        key = tag.lower().strip()
        if not key or key in seen_lower:
            continue
        seen_lower.add(key)
        entries.append({
            "index": len(entries),
            "id": slugify(tag),
            "label": tag,
            "group": "general",
            "prompt": f"a photo of {tag}",
        })

    # 2. Add specialized SAN-1 categories if not already represented by exact label
    for cat in san1_cats:
        label = cat["label"]
        key = label.lower().strip()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        prompts = cat.get("prompts", [])
        primary_prompt = prompts[0] if prompts else f"a photo of {label}"
        entries.append({
            "index": len(entries),
            "id": slugify(label),
            "label": label,
            "group": cat.get("group", "specialized"),
            "prompt": primary_prompt,
        })

    vocab_data = {
        "version": "1.0.0",
        "name": "SAN-Web Multi-Label English Visual Tag Vocabulary",
        "num_classes": len(entries),
        "default_threshold": 0.50,
        "recommended_top_k": 15,
        "labels": [e["label"] for e in entries],
        "tags": entries,
    }

    labels_json_path = MODELS_DIR / "labels.json"
    labels_txt_path = MODELS_DIR / "labels.txt"

    labels_json_path.write_text(json.dumps(vocab_data, indent=2, ensure_ascii=False), encoding="utf-8")
    labels_txt_path.write_text("\n".join(e["label"] for e in entries) + "\n", encoding="utf-8")

    # Clean up temporary raw cache if desired
    raw_cache = MODELS_DIR / "raw_ram_tags.txt"
    if raw_cache.exists():
        raw_cache.unlink()

    print(f"[vocab] Built vocabulary with {len(entries)} English tags.")
    print(f"[vocab] Saved -> {labels_json_path} ({labels_json_path.stat().st_size / 1024:.1f} KB)")
    print(f"[vocab] Saved -> {labels_txt_path} ({labels_txt_path.stat().st_size / 1024:.1f} KB)")
    return vocab_data


if __name__ == "__main__":
    build_vocabulary()

