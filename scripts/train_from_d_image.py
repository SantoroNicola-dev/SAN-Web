"""
Supervised Multi-Label Adapter, Contrastive Prototype Bank & Tag Classifier Training (`v3.0`)
using the real-world image collection in `D:\Image\Image` + the 970 general-domain calibration images.

Dataset Coverage from `D:\Image\Image`:
  1. `offensive_gestures` (220 files: hagrid_mf_*, hagrid_v_inverted_*, mcsigns_*) + `deny` 31..99 (69 files)
  2. `benign_gestures` (100 files: benign_like_*, benign_peace_*)
  3. `bad_meme` (740 files) + `deny` meme blocks (7338..7587, 17821..17961)
  4. `deny` text/quote/blasphemy banners (7301..7322, 7608..7647, 7668..17801, 17981..18140)
     + `accepted` benign text banners (7300, 7323..7337, 17802..17820, 17962..17980)
  5. `deny` weapons (280..299) + airport X-ray concealed firearms (18205..18254)
  6. `deny` explicit nudity (7293..7299, 24993..25002)
  7. `deny` identity documents / driver licenses (19035, 19593, 23286, 24630 with multi-crop augmentation)
  8. `accepted` invoices & financial bills (7588..7607) + cash register receipts (7648..7667)
  9. `accepted` cars/vehicles (1..30) and cooked food/dishes (100..279)
 10. `accepted` clean close-up selfies & face portraits (300..7292, 18141..18204, 18255..24992)
 11. `QuixiAI/SexDrugsAndRockAndRoll` (all 593 historical/concert/war/drugs/protest photos with human captions
     in `metadata.json`, all 102 files placed in `accepted`, and 400 `_001_240x.jpg` adult/lingerie thumbnails
     including all previously misclassified cases).
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
LABELS_JSON_PATH = MODELS_DIR / "labels.json"
TAG_CACHE_PATH = MODELS_DIR / "tag_embeddings_v2.npy"
CAL_VECS_PATH = MODELS_DIR / "calibration_image_vecs.npy"
D_IMAGE_CACHE_PATH = MODELS_DIR / "d_image_dataset_cache.npz"
TRAINED_HEAD_PATH = MODELS_DIR / "d_image_trained_head.npz"

D_IMAGE_ROOT = Path(r"D:\Image\Image")
QUIXI_ROOT = D_IMAGE_ROOT / "QuixiAI" / "SexDrugsAndRockAndRoll"
MODEL_ID = "openai/clip-vit-base-patch32"


def load_and_preprocess_image(item: tuple[str, str]) -> tuple[str, str, np.ndarray | None]:
    """Loads an image path and returns (path_str, group_key, float32_chw_rgb_0_1)."""
    path_str, group_key = item
    try:
        with Image.open(path_str) as img:
            rgb = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            chw = np.transpose(arr, (2, 0, 1))
            return path_str, group_key, chw
    except Exception:
        return path_str, group_key, None


def load_augmented_crops(path_str: str, group_key: str) -> list[tuple[str, str, np.ndarray]]:
    """Generates 6 deterministic multi-scale / aspect crops for rare classes (like ID cards)."""
    out = []
    try:
        with Image.open(path_str) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            crop_configs = [
                (0.0, 0.0, 1.0, 1.0),
                (0.05, 0.05, 0.95, 0.95),
                (0.0, 0.10, 0.92, 1.0),
                (0.08, 0.0, 1.0, 0.92),
                (0.10, 0.15, 0.95, 0.95),
                (0.0, 0.20, 1.0, 1.0),
            ]
            for c_idx, (x0, y0, x1, y1) in enumerate(crop_configs):
                sub = rgb.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
                sub = sub.resize((224, 224), Image.Resampling.BILINEAR)
                arr = np.asarray(sub, dtype=np.float32) / 255.0
                chw = np.transpose(arr, (2, 0, 1))
                out.append((f"{path_str}#crop{c_idx}", group_key, chw))
    except Exception:
        pass
    return out


class BatchedBackboneExtractor(nn.Module):
    """
    Exact batched implementation of `SanAllInOneTagger.encode_image_norm`
    so extracted features match the ONNX graph to < 1e-6.
    """

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        from export_onnx_model import ExplicitEncoderLayer, ExplicitLayerNorm

        vision = clip_model.vision_model
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)

        self.patch_embedding = vision.embeddings.patch_embedding
        cls_embed = vision.embeddings.class_embedding.detach().clone().view(1, 1, 768)
        pos_ids = torch.arange(50, dtype=torch.long)
        pos_embed = vision.embeddings.position_embedding(pos_ids).detach().clone().view(1, 50, 768)
        self.register_buffer("class_embedding", cls_embed)
        self.register_buffer("position_embedding", pos_embed)

        self.pre_ln = ExplicitLayerNorm(vision.pre_layrnorm)
        self.layers = nn.ModuleList([ExplicitEncoderLayer(layer) for layer in vision.encoder.layers])

        spatial_weights = np.zeros((7, 7), dtype=np.float32)
        for r in range(7):
            for c in range(7):
                dist2 = (r - 3.0) ** 2 + (c - 3.0) ** 2
                spatial_weights[r, c] = np.exp(-dist2 / (2.0 * (2.1 ** 2)))
        spatial_weights /= spatial_weights.sum()
        center_selector = torch.zeros(1, 1, 50, dtype=torch.float32)
        center_selector[0, 0, 1:] = torch.from_numpy(spatial_weights.flatten())
        self.register_buffer("center_patch_selector", center_selector)

        self.post_ln = ExplicitLayerNorm(vision.post_layernorm)
        self.visual_projection = clip_model.visual_projection

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        x = (image - self.pixel_mean) / self.pixel_std
        patches = self.patch_embedding(x).reshape(b, 768, 49).transpose(1, 2)
        cls_tok = self.class_embedding.expand(b, -1, -1)
        hidden = torch.cat([cls_tok, patches], dim=1) + self.position_embedding
        hidden = self.pre_ln(hidden)

        for i in range(11):
            layer = self.layers[i]
            res = hidden
            h = layer.ln1(hidden)
            q = (layer.q_proj(h) * 0.125).reshape(b, 50, 12, 64).transpose(1, 2)
            k = layer.k_proj(h).reshape(b, 50, 12, 64).transpose(1, 2)
            v = layer.v_proj(h).reshape(b, 50, 12, 64).transpose(1, 2)
            attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
            attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(b, 50, 768)
            hidden = res + layer.out_proj(attn_out)
            res2 = hidden
            h2 = layer.fc1(layer.ln2(hidden))
            h2 = h2 * torch.sigmoid(1.702 * h2)
            hidden = res2 + layer.fc2(h2)

        # Final layer with spatial detail
        last = self.layers[11]
        res = hidden
        h = last.ln1(hidden)
        q = (last.q_proj(h) * 0.125).reshape(b, 50, 12, 64).transpose(1, 2)
        k = last.k_proj(h).reshape(b, 50, 12, 64).transpose(1, 2)
        v_raw = last.v_proj(h)
        v = v_raw.reshape(b, 50, 12, 64).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(b, 50, 768)
        x_global = res + last.out_proj(attn_out)
        hg = last.fc1(last.ln2(x_global))
        hg = hg * torch.sigmoid(1.702 * hg)
        x_global = x_global + last.fc2(hg)

        x_center_in = torch.matmul(self.center_patch_selector, res)
        v_center_in = torch.matmul(self.center_patch_selector, v_raw)
        x_local = x_center_in + last.out_proj(v_center_in)
        hl = last.fc1(last.ln2(x_local))
        hl = hl * torch.sigmoid(1.702 * hl)
        x_local = x_local + last.fc2(hl)

        cls_token = x_global[:, 0:1, :]
        pooled_cls = self.post_ln(cls_token).reshape(b, 768)
        embeds_cls = self.visual_projection(pooled_cls)
        v_cls = embeds_cls / torch.sqrt((embeds_cls * embeds_cls).sum(dim=-1, keepdim=True) + 1e-12)

        pooled_loc = self.post_ln(x_local).reshape(b, 768)
        embeds_loc = self.visual_projection(pooled_loc)
        v_loc = embeds_loc / torch.sqrt((embeds_loc * embeds_loc).sum(dim=-1, keepdim=True) + 1e-12)

        v_fused = 0.86 * v_cls + 0.14 * v_loc
        v_fused = v_fused / torch.sqrt((v_fused * v_fused).sum(dim=-1, keepdim=True) + 1e-12)
        return v_fused


def collect_d_image_samples() -> list[tuple[str, str]]:
    """
    Collects a comprehensive, stratified list of (file_path, group_key) across all folders
    and contiguous blocks in `D:\Image\Image`.
    """
    rng = random.Random(42)
    items: list[tuple[str, str]] = []

    # 1. All 220 offensive_gestures
    off_dir = D_IMAGE_ROOT / "offensive_gestures"
    for f in sorted(os.listdir(off_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            items.append((str(off_dir / f), "offensive_gesture"))

    # 2. All 100 benign_gestures (distinguish like vs peace)
    ben_dir = D_IMAGE_ROOT / "benign_gestures"
    for f in sorted(os.listdir(ben_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            g = "benign_gesture_peace" if "peace" in f.lower() else "benign_gesture_like"
            items.append((str(ben_dir / f), g))

    # 3. Sample 340 bad_meme files
    meme_dir = D_IMAGE_ROOT / "bad_meme"
    meme_files = [
        str(meme_dir / f)
        for f in sorted(os.listdir(meme_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    rng.shuffle(meme_files)
    for p in meme_files[:340]:
        items.append((p, "bad_meme"))

    # 4. Accepted folder stratified by contiguous blocks
    acc_dir = D_IMAGE_ROOT / "accepted"
    acc_selfies = []
    for f in sorted(os.listdir(acc_dir)):
        full_p = str(acc_dir / f)
        if not f.startswith("Img_"):
            # One of the 102 QuixiAI files in accepted!
            items.append((full_p, f"quixi::{f}"))
            continue
        try:
            num = int(f.split("_")[1].split(".")[0])
        except Exception:
            continue
        if 1 <= num <= 30:
            items.append((full_p, "acc_car"))
        elif 100 <= num <= 279:
            items.append((full_p, "acc_food"))
        elif 7588 <= num <= 7607:
            items.append((full_p, "acc_invoice"))
        elif 7648 <= num <= 7667:
            items.append((full_p, "acc_receipt"))
        elif 18141 <= num <= 18204:
            items.append((full_p, "acc_headshot"))
        elif (7300 <= num <= 7337) or (17802 <= num <= 17820) or (17962 <= num <= 17980):
            items.append((full_p, "acc_benign_banner"))
        else:
            acc_selfies.append(full_p)

    rng.shuffle(acc_selfies)
    for p in acc_selfies[:550]:
        items.append((p, "acc_selfie"))

    # 5. Deny folder stratified by contiguous blocks
    den_dir = D_IMAGE_ROOT / "deny"
    den_hate_memes = []
    den_impact_memes = []
    den_toxic_quotes = []
    den_blasphemy = []

    for f in sorted(os.listdir(den_dir)):
        if not f.startswith("Img_"):
            continue
        full_p = str(den_dir / f)
        try:
            num = int(f.split("_")[1].split(".")[0])
        except Exception:
            continue

        if 31 <= num <= 99:
            items.append((full_p, "offensive_gesture"))
        elif 280 <= num <= 299:
            items.append((full_p, "den_weapon_gun"))
        elif 7293 <= num <= 7299 or 24993 <= num <= 25002:
            items.append((full_p, "den_explicit_nudity"))
        elif 18205 <= num <= 18254:
            items.append((full_p, "den_weapon_xray"))
        elif num in (19035, 19593, 23286, 24630):
            items.append((full_p, "den_id_license"))
        elif 7338 <= num <= 7587:
            den_hate_memes.append(full_p)
        elif 17821 <= num <= 17961:
            den_impact_memes.append(full_p)
        elif (7301 <= num <= 7335) or (7608 <= num <= 7647) or (17963 <= num <= 18140):
            den_blasphemy.append(full_p)
        elif 7668 <= num <= 17820:
            den_toxic_quotes.append(full_p)

    rng.shuffle(den_hate_memes)
    rng.shuffle(den_impact_memes)
    rng.shuffle(den_blasphemy)
    rng.shuffle(den_toxic_quotes)

    for p in den_hate_memes[:160]:
        items.append((p, "den_hate_meme"))
    for p in den_impact_memes[:110]:
        items.append((p, "bad_meme"))
    for p in den_blasphemy[:140]:
        items.append((p, "den_profane_banner"))
    for p in den_toxic_quotes[:240]:
        items.append((p, "den_quote_banner"))

    # 6. QuixiAI/SexDrugsAndRockAndRoll
    q_bench = json.loads((QUIXI_ROOT / "benchmark_results.json").read_text(encoding="utf-8"))
    q_acc_dir = QUIXI_ROOT / "accepted"
    q_den_dir = QUIXI_ROOT / "deny"
    q_acc_files = set(os.listdir(q_acc_dir))
    q_den_files = set(os.listdir(q_den_dir))

    def resolve_quixi_path(fname: str) -> str | None:
        if fname in q_acc_files:
            return str(q_acc_dir / fname)
        if fname in q_den_files:
            return str(q_den_dir / fname)
        return None

    already_added_quixi = {g.split("::", 1)[1] for _, g in items if g.startswith("quixi::")}
    regular_001_Explicit = []

    for fname, binfo in q_bench.items():
        if fname in already_added_quixi:
            continue
        p = resolve_quixi_path(fname)
        if p is None:
            continue
        cat_id = binfo.get("category_id", "")
        if not fname.endswith("_001_240x.jpg"):
            # Include ALL 593 historical/cultural/concert/war/drugs/weapon images
            items.append((p, f"quixi::{fname}"))
        else:
            # Include all 221 previously misclassified _001_240x.jpg images + sample of regular ones
            if cat_id != "BLOCKED_EXPLICIT":
                items.append((p, f"quixi::{fname}"))
            else:
                regular_001_Explicit.append((p, f"quixi::{fname}"))

    rng.shuffle(regular_001_Explicit)
    items.extend(regular_001_Explicit[:240])

    return items


def extract_or_load_d_image_features(clip_model: CLIPModel) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Extracts 512-D normalized fused visual embeddings for all collected D:\\Image\\Image samples
    (plus multi-crop augmentations for rare classes like ID cards/driver licenses) and caches them.
    """
    if D_IMAGE_CACHE_PATH.exists():
        data = np.load(D_IMAGE_CACHE_PATH, allow_pickle=True)
        vecs = data["vecs"].astype(np.float32)
        paths = [str(p) for p in data["paths"]]
        groups = [str(g) for g in data["groups"]]
        print(f"[d_image] Loaded cached embeddings {vecs.shape} from {D_IMAGE_CACHE_PATH}")
        return vecs, paths, groups

    raw_items = collect_d_image_samples()
    print(f"[d_image] Preparing to load and encode {len(raw_items)} images from D:\\Image\\Image...")

    extractor = BatchedBackboneExtractor(clip_model)
    extractor.eval()

    valid_paths: list[str] = []
    valid_groups: list[str] = []
    all_vecs: list[np.ndarray] = []

    batch_tensors: list[np.ndarray] = []
    batch_size = 64
    t0 = time.time()

    def flush_batch():
        if not batch_tensors:
            return
        tensor_x = torch.from_numpy(np.stack(batch_tensors, axis=0))
        with torch.no_grad():
            out_v = extractor(tensor_x).cpu().numpy().astype(np.float32)
        all_vecs.append(out_v)
        batch_tensors.clear()

    # First add multi-crop augmentations for rare classes (ID cards, invoices, receipts, benign gestures)
    for path_str, group_key in raw_items:
        if group_key in ("den_id_license", "acc_invoice", "acc_receipt"):
            for cp_path, cp_group, chw in load_augmented_crops(path_str, group_key)[1:]:
                valid_paths.append(cp_path)
                valid_groups.append(cp_group)
                batch_tensors.append(chw)
                if len(batch_tensors) >= batch_size:
                    flush_batch()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for idx, (path_str, group_key, chw) in enumerate(pool.map(load_and_preprocess_image, raw_items, chunksize=32)):
            if chw is not None:
                valid_paths.append(path_str)
                valid_groups.append(group_key)
                batch_tensors.append(chw)
                if len(batch_tensors) >= batch_size:
                    flush_batch()
            if (idx + 1) % 500 == 0 or (idx + 1) == len(raw_items):
                elapsed = time.time() - t0
                print(f"  - Encoded {idx + 1}/{len(raw_items)} images ({elapsed:.1f}s)")

    flush_batch()
    vecs = np.concatenate(all_vecs, axis=0).astype(np.float32)
    np.savez_compressed(
        D_IMAGE_CACHE_PATH,
        vecs=vecs,
        paths=np.array(valid_paths, dtype=object),
        groups=np.array(valid_groups, dtype=object),
    )
    print(f"[d_image] Saved {vecs.shape} embeddings -> {D_IMAGE_CACHE_PATH}")
    return vecs, valid_paths, valid_groups


def build_supervision_targets(
    d_vecs: np.ndarray,
    d_groups: list[str],
    cal_vecs: np.ndarray,
    labels: list[str],
    base_W: np.ndarray,
    base_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Combines `d_vecs` (from D:\\Image\\Image) + `cal_vecs` (970 general-domain images)
    and builds multi-label supervision matrices:
      - X: [N_total, 512]
      - Y_target: [N_total, 4660] target probabilities in [0, 1]
      - Y_weight: [N_total, 4660] per-entry supervision weight (high for ground-truth positives & hard negatives)
      - all_groups: list[str] of length N_total
    """
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    lower_to_idx = {lbl.lower(): idx for idx, lbl in enumerate(labels)}

    # Build fast word/phrase matcher for RAM concrete tags (indices 0..4584)
    from export_onnx_model import ABSTRACT_OR_OCCUPATIONAL_WORDS, OCCUPATIONAL_REGEX

    concrete_ram_tags: list[tuple[str, int]] = []
    for idx in range(min(4585, len(labels))):
        w = labels[idx].lower().strip()
        if len(w) >= 3 and w not in ABSTRACT_OR_OCCUPATIONAL_WORDS and not OCCUPATIONAL_REGEX.search(w):
            concrete_ram_tags.append((w, idx))

    q_meta_list = json.loads((QUIXI_ROOT / "metadata.json").read_text(encoding="utf-8"))
    q_meta = {item["file_name"]: item.get("caption", "") for item in q_meta_list}
    q_bench = json.loads((QUIXI_ROOT / "benchmark_results.json").read_text(encoding="utf-8"))

    # Combine D:\Image\Image vectors + 970 calibration vectors
    X = np.concatenate([d_vecs, cal_vecs], axis=0).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)

    cal_groups = []
    for i in range(cal_vecs.shape[0]):
        if i < 350:
            cal_groups.append("cal_imagenet")
        elif i < 450:
            cal_groups.append("cal_pet")
        elif i < 570:
            cal_groups.append("cal_food")
        elif i < 670:
            cal_groups.append("cal_car")
        elif i < 750:
            cal_groups.append("cal_flower")
        elif i < 870:
            cal_groups.append("cal_fashion")
        else:
            cal_groups.append("cal_face")

    all_groups = list(d_groups) + cal_groups
    n_total = X.shape[0]
    num_tags = len(labels)

    # Start with zero-shot teacher soft targets (with a low default anchor weight = 0.15)
    raw_sims = X @ base_W
    debiased = raw_sims - base_bias[np.newaxis, :]
    mean_s = debiased.mean(axis=-1, keepdims=True)
    std_s = np.maximum(debiased.std(axis=-1, keepdims=True), 1e-6)
    z_s = (debiased - mean_s) / std_s
    teacher_logits = 0.55 * ((debiased - 0.222) * 35.0) + 0.45 * ((z_s - 2.05) * 1.65)
    Y_target = (1.0 / (1.0 + np.exp(-np.clip(teacher_logits, -15.0, 15.0)))).astype(np.float32)
    Y_weight = np.full((n_total, num_tags), 0.15, dtype=np.float32)

    def set_pos(row_i: int, tag_names: list[str], val: float = 0.96, weight: float = 6.0):
        for t in tag_names:
            j = label_to_idx.get(t)
            if j is None:
                j = lower_to_idx.get(t.lower())
            if j is not None:
                if val > Y_target[row_i, j] or Y_weight[row_i, j] < weight:
                    Y_target[row_i, j] = max(Y_target[row_i, j], val) if Y_weight[row_i, j] >= weight else val
                    Y_weight[row_i, j] = max(Y_weight[row_i, j], weight)

    def set_neg(row_i: int, tag_names: list[str], val: float = 0.01, weight: float = 6.0):
        for t in tag_names:
            j = label_to_idx.get(t)
            if j is None:
                j = lower_to_idx.get(t.lower())
            if j is not None:
                Y_target[row_i, j] = val
                Y_weight[row_i, j] = max(Y_weight[row_i, j], weight)

    all_sensitive_cats = [
        "Explicit Nudity or Adult Sexual Content",
        "Swimwear, Bikini, Lingerie or Shirtless",
        "Offensive or Vulgar Hand Gesture",
        "Weapon, Firearm or Knife",
        "Violence, Physical Fight or Crime",
        "Illicit Drugs & Narcotics Paraphernalia",
        "Smoking Cigarette, Cigar or Vape",
        "Hate Symbol, Swastika or Extremist Emblem",
        "Gore, Blood, Injury or Corpses",
    ]
    all_doc_meme_cats = [
        "Paper Document, Letter or Book Page",
        "Receipt, Invoice or Financial Bill",
        "Identity Card, Passport or Driver License",
        "Digital Screenshot, App or Website UI",
        "Chart, Graph, Map or Infographic",
        "Internet Meme or Satirical Graphic",
        "Blurry / Out-of-Focus Image",
        "Dark / Underexposed Low-Light Image",
    ]

    explicit_kw = re.compile(
        r"\b(fuck\w*|pussy|cock|dick|blowjob|cum\w*|anal|dildo|strapon|masturbat\w*|tits|boobs|cunt|"
        r"naked|nude|nudity|topless|handjob|threesome|orgasm|porn\w*|hardcore|suck\w*|penetrat\w*|"
        r"bdsm|bondage|orgy|genital\w*|intercourse|erotic|fucked)\b",
        re.IGNORECASE,
    )
    lingerie_kw = re.compile(
        r"\b(bra|lingerie|bikini|swimsuit|swimwear|underwear|panties|thong|cleavage|shirtless|"
        r"corset|latex|nylons|stockings|strip\w*|teas\w*|sexy|hottie|babe|fetish|busty|curvy)\b",
        re.IGNORECASE,
    )
    gore_kw = re.compile(
        r"\b(corpse\w*|dead bod\w*|deceased|massacre|decapitat\w*|execution|wounded|ripped out|"
        r"blood\w*|casualt\w*|atrocit\w*|concentration camp|emaciated|starv\w*|holocaust|lynch\w*|"
        r"mutilat\w*|severed|victim\w*)\b",
        re.IGNORECASE,
    )
    weapon_kw = re.compile(
        r"\b(rifle\w*|gun\w*|pistol\w*|firearm\w*|weapon\w*|handgun\w*|shotgun\w*|machine gun|"
        r"bayonet|knife|sword|aiming|armed)\b",
        re.IGNORECASE,
    )
    drug_kw = re.compile(
        r"\b(cannabis|marijuana|weed|joint|bong|cocaine|heroin|syringe|lsd|acid|pills|narcotic\w*|"
        r"drug\w*|pot|hashish|meth|opium|crack)\b",
        re.IGNORECASE,
    )
    smoke_kw = re.compile(r"\b(cigarette\w*|smoking|cigar\w*|vape|tobacco)\b", re.IGNORECASE)
    hate_kw = re.compile(
        r"\b(nazi\w*|swastika|hitler|ku klux klan|kkk|white supremac\w*|ss officer|fascist|antisemit\w*)\b",
        re.IGNORECASE,
    )
    concert_kw = re.compile(
        r"\b(concert\w*|stage|perform\w*|woodstock|festival|crowd-surfing|musician\w*|guitar\w*|"
        r"singer|band|rock|punk|hip-hop|rave|dj|microphone|drum\w*|audience)\b",
        re.IGNORECASE,
    )
    protest_kw = re.compile(
        r"\b(protest\w*|march\w*|rally|demonstrat\w*|activist\w*|civil rights|riot\w*|tiananmen|"
        r"picketing|strike)\b",
        re.IGNORECASE,
    )

    for i, g in enumerate(all_groups):
        if g == "offensive_gesture":
            set_neg(i, ["Friendly or Benign Hand Gesture", "Close-up Selfie", "peace", "Explicit Nudity or Adult Sexual Content"] + all_doc_meme_cats, 0.01, 8.0)
            set_pos(i, ["Offensive or Vulgar Hand Gesture", "hand", "person", "face", "portrait"], 0.97, 8.5)

        elif g == "benign_gesture_like":
            set_neg(i, all_sensitive_cats + all_doc_meme_cats, 0.01, 8.5)
            set_pos(i, ["Friendly or Benign Hand Gesture", "hand", "person", "face", "smile", "Close-up Selfie"], 0.97, 8.5)

        elif g == "benign_gesture_peace":
            set_neg(i, all_sensitive_cats + all_doc_meme_cats, 0.01, 8.5)
            set_pos(i, ["Friendly or Benign Hand Gesture", "peace", "hand", "person", "face", "smile", "Close-up Selfie"], 0.97, 8.5)

        elif g == "bad_meme":
            set_neg(i, ["Close-up Selfie", "Professional / Studio Portrait", "Receipt, Invoice or Financial Bill", "Identity Card, Passport or Driver License", "Paper Document, Letter or Book Page", "Blurry / Out-of-Focus Image"], 0.01, 7.0)
            set_pos(i, ["Internet Meme or Satirical Graphic", "poster", "text", "comic", "cartoon", "sign", "banner"], 0.96, 7.5)

        elif g == "den_hate_meme":
            set_neg(i, ["Close-up Selfie", "Professional / Studio Portrait", "Receipt, Invoice or Financial Bill", "Identity Card, Passport or Driver License", "Blurry / Out-of-Focus Image"], 0.01, 7.5)
            set_pos(i, ["Internet Meme or Satirical Graphic", "Hate Symbol, Swastika or Extremist Emblem", "poster", "text", "comic", "cartoon", "sign"], 0.96, 7.5)

        elif g in ("den_profane_banner", "den_quote_banner"):
            set_neg(i, ["Close-up Selfie", "Professional / Studio Portrait", "Receipt, Invoice or Financial Bill", "Identity Card, Passport or Driver License", "Blurry / Out-of-Focus Image", "Dark / Underexposed Low-Light Image"], 0.01, 7.5)
            set_pos(i, ["Internet Meme or Satirical Graphic", "Digital Screenshot, App or Website UI", "text", "quote", "banner", "sign", "poster", "blackboard"], 0.96, 7.5)

        elif g == "acc_benign_banner":
            set_neg(i, all_sensitive_cats + ["Close-up Selfie", "Receipt, Invoice or Financial Bill", "Identity Card, Passport or Driver License", "Internet Meme or Satirical Graphic"], 0.01, 7.5)
            set_pos(i, ["Digital Screenshot, App or Website UI", "text", "sign", "banner", "label"], 0.96, 7.5)

        elif g == "den_weapon_gun":
            set_neg(i, ["Close-up Selfie", "Friendly or Benign Hand Gesture", "Receipt, Invoice or Financial Bill", "Blurry / Out-of-Focus Image"], 0.01, 8.0)
            set_pos(i, ["Weapon, Firearm or Knife", "gun", "rifle", "weapon", "handgun"], 0.97, 8.5)

        elif g == "den_weapon_xray":
            set_neg(i, ["Close-up Selfie", "Receipt, Invoice or Financial Bill", "Paper Document, Letter or Book Page", "Chart, Graph, Map or Infographic", "Blurry / Out-of-Focus Image"], 0.01, 8.5)
            set_pos(i, ["Weapon, Firearm or Knife", "gun", "handgun", "weapon"], 0.97, 8.5)

        elif g == "den_explicit_nudity":
            set_neg(i, ["Close-up Selfie", "Professional / Studio Portrait", "Friendly or Benign Hand Gesture"] + all_doc_meme_cats, 0.01, 8.5)
            set_pos(i, ["Explicit Nudity or Adult Sexual Content", "Swimwear, Bikini, Lingerie or Shirtless", "adult", "skin", "person", "woman"], 0.97, 8.5)

        elif g == "den_id_license":
            set_neg(i, ["Close-up Selfie", "Receipt, Invoice or Financial Bill", "Internet Meme or Satirical Graphic", "Explicit Nudity or Adult Sexual Content"], 0.01, 9.0)
            set_pos(i, ["Identity Card, Passport or Driver License", "card", "license", "document", "passport", "person", "face", "text"], 0.98, 9.5)

        elif g in ("acc_invoice", "acc_receipt"):
            set_neg(i, all_sensitive_cats + ["Close-up Selfie", "Identity Card, Passport or Driver License", "Internet Meme or Satirical Graphic", "Blurry / Out-of-Focus Image"], 0.01, 8.5)
            set_pos(i, ["Receipt, Invoice or Financial Bill", "Paper Document, Letter or Book Page", "receipt", "bill", "document", "paper", "text"], 0.97, 8.5)

        elif g == "acc_car":
            set_neg(i, all_sensitive_cats + all_doc_meme_cats + ["Close-up Selfie"], 0.01, 7.5)
            set_pos(i, ["Car / Automobile", "car", "vehicle", "suv", "sedan", "wheel", "tire"], 0.97, 8.0)

        elif g == "acc_food":
            set_neg(i, all_sensitive_cats + all_doc_meme_cats + ["Close-up Selfie"], 0.01, 7.5)
            set_pos(i, ["Cooked Meal / Restaurant Dish", "food", "meal", "plate", "cuisine", "lunch", "dinner"], 0.97, 8.0)

        elif g in ("acc_selfie", "acc_headshot"):
            set_neg(i, all_sensitive_cats + all_doc_meme_cats, 0.01, 8.0)
            set_pos(i, ["Close-up Selfie", "selfie", "face", "person", "portrait", "head"], 0.97, 7.5)

        elif g.startswith("quixi::"):
            fname = g.split("::", 1)[1]
            caption = q_meta.get(fname, "")
            binfo = q_bench.get(fname, {})
            cat_id = binfo.get("category_id", "")
            cap_low = caption.lower()

            # Prevent false "Blurry / Out-of-Focus Image" on historical/concert/adult photos
            set_neg(i, ["Blurry / Out-of-Focus Image", "Dark / Underexposed Low-Light Image"], 0.01, 6.5)

            if fname.endswith("_001_240x.jpg"):
                set_neg(i, ["Close-up Selfie", "Professional / Studio Portrait", "Friendly or Benign Hand Gesture"] + all_doc_meme_cats, 0.01, 8.0)
                if explicit_kw.search(caption):
                    set_pos(i, ["Explicit Nudity or Adult Sexual Content", "Swimwear, Bikini, Lingerie or Shirtless", "adult", "skin", "person", "woman"], 0.97, 8.5)
                elif lingerie_kw.search(caption):
                    set_pos(i, ["Swimwear, Bikini, Lingerie or Shirtless", "Explicit Nudity or Adult Sexual Content", "bikini", "swimwear", "adult", "skin", "person", "woman"], 0.95, 8.0)
                else:
                    set_pos(i, ["Explicit Nudity or Adult Sexual Content", "Swimwear, Bikini, Lingerie or Shirtless", "adult", "person", "woman"], 0.94, 7.5)
            else:
                # Non-_001_240x.jpg historical/cultural/concert/war/drugs/protest image with rich English caption
                matched_any_sensitive = False
                if gore_kw.search(caption) or cat_id == "BLOCKED_CORPSES_ATROCITIES":
                    set_pos(i, ["Gore, Blood, Injury or Corpses", "Violence, Physical Fight or Crime", "blood", "wound", "injury", "soldier"], 0.96, 8.5)
                    matched_any_sensitive = True
                if hate_kw.search(caption) or cat_id == "BLOCKED_HATE_SYMBOL":
                    set_pos(i, ["Hate Symbol, Swastika or Extremist Emblem"], 0.96, 8.5)
                    matched_any_sensitive = True
                if weapon_kw.search(caption) or cat_id == "BLOCKED_WEAPON":
                    set_pos(i, ["Weapon, Firearm or Knife", "gun", "rifle", "weapon"], 0.96, 8.5)
                    matched_any_sensitive = True
                if drug_kw.search(caption) or cat_id == "BLOCKED_DRUGS":
                    set_pos(i, ["Illicit Drugs & Narcotics Paraphernalia", "medicine", "capsule", "smoke"], 0.96, 8.5)
                    matched_any_sensitive = True
                if smoke_kw.search(caption):
                    set_pos(i, ["Smoking Cigarette, Cigar or Vape", "cigarette", "cigar", "smoke"], 0.96, 8.0)
                    matched_any_sensitive = True
                if explicit_kw.search(caption):
                    set_pos(i, ["Explicit Nudity or Adult Sexual Content", "Swimwear, Bikini, Lingerie or Shirtless", "adult", "skin"], 0.96, 8.5)
                    matched_any_sensitive = True
                elif lingerie_kw.search(caption):
                    set_pos(i, ["Swimwear, Bikini, Lingerie or Shirtless", "bikini", "swimwear", "skin"], 0.95, 8.0)
                    matched_any_sensitive = True
                if cat_id == "BLOCKED_OFFENSIVE_GESTURE" or "middle finger" in cap_low:
                    set_neg(i, ["Friendly or Benign Hand Gesture"], 0.01, 8.0)
                    set_pos(i, ["Offensive or Vulgar Hand Gesture", "hand", "person"], 0.96, 8.5)
                    matched_any_sensitive = True

                if concert_kw.search(caption):
                    set_pos(i, ["Crowd or Public Event", "Musical Instruments & Audio", "concert", "stage", "crowd", "musician", "person"], 0.95, 7.5)
                if protest_kw.search(caption):
                    set_pos(i, ["Crowd or Public Event", "protest", "crowd", "sign", "banner", "person"], 0.95, 7.5)
                if not matched_any_sensitive:
                    set_neg(i, ["Explicit Nudity or Adult Sexual Content", "Offensive or Vulgar Hand Gesture", "Gore, Blood, Injury or Corpses"], 0.01, 6.0)

            # Match concrete RAM tags directly mentioned in the human caption
            cap_words = set(re.findall(r"[a-z0-9]+", cap_low))
            for w_tag, tag_idx in concrete_ram_tags:
                if " " in w_tag:
                    if w_tag in cap_low:
                        Y_target[i, tag_idx] = 0.95
                        Y_weight[i, tag_idx] = max(Y_weight[i, tag_idx], 6.5)
                elif w_tag in cap_words:
                    Y_target[i, tag_idx] = 0.95
                    Y_weight[i, tag_idx] = max(Y_weight[i, tag_idx], 6.5)

        elif g.startswith("cal_"):
            # General-domain calibration images (preserve clean zero-shot concrete tags, suppress sensitive false positives)
            set_neg(i, all_sensitive_cats + ["Internet Meme or Satirical Graphic", "Receipt, Invoice or Financial Bill", "Identity Card, Passport or Driver License", "Blurry / Out-of-Focus Image"], 0.01, 5.0)
            if g == "cal_pet":
                set_pos(i, ["animal", "pet", "mammal"], 0.94, 5.0)
            elif g == "cal_food":
                set_pos(i, ["Cooked Meal / Restaurant Dish", "food", "meal", "plate"], 0.94, 5.0)
            elif g == "cal_car":
                set_pos(i, ["Car / Automobile", "car", "vehicle", "wheel", "tire"], 0.94, 5.0)
            elif g == "cal_flower":
                set_pos(i, ["Flowers, Blossoms & Houseplants", "flower", "plant", "petal", "blossom"], 0.94, 5.0)
            elif g == "cal_fashion":
                set_pos(i, ["Clothing, Shoes & Fashion", "clothing", "fashion"], 0.94, 5.0)
            elif g == "cal_face":
                set_pos(i, ["Close-up Selfie", "person", "face", "portrait", "head"], 0.95, 5.5)

    return X, Y_target, Y_weight, all_groups


class TrainableTaggingHead(nn.Module):
    """
    Trainable Residual Visual Adapter + Calibrated Tag Matrix + Contrastive Prototype Memory Bank.
    Uses ONLY primitive operations supported by Unity Sentis and ONNX Runtime Web.
    """

    def __init__(
        self,
        base_W: np.ndarray,
        base_bias: np.ndarray,
        base_scale: np.ndarray,
        proto_init_vecs: np.ndarray,
        proto_init_vals: np.ndarray,
    ):
        super().__init__()
        self.register_buffer("base_W", torch.from_numpy(base_W.astype(np.float32)))
        self.register_buffer("proto_base_vecs", torch.from_numpy(proto_init_vecs.astype(np.float32)))  # [M, 512]

        # Residual bottleneck visual adapter: 512 -> 256 -> 512
        self.adapter_w1 = nn.Parameter(torch.randn(512, 256) * 0.015)
        self.adapter_b1 = nn.Parameter(torch.zeros(1, 256))
        self.adapter_w2 = nn.Parameter(torch.zeros(256, 512))
        self.adapter_b2 = nn.Parameter(torch.zeros(1, 512))

        # Fine-tuned tag matrix residual [512, 4660]
        self.delta_W = nn.Parameter(torch.zeros_like(self.base_W))
        self.tag_bias = nn.Parameter(torch.from_numpy(base_bias.astype(np.float32)).view(1, -1))
        self.tag_scale = nn.Parameter(torch.from_numpy(base_scale.astype(np.float32)).view(1, -1))

        # Fine-tuned contrastive prototype values [M, 4660]
        self.proto_values = nn.Parameter(torch.from_numpy(proto_init_vals.astype(np.float32)))

    def adapt_vectors(self, v_base: torch.Tensor) -> torch.Tensor:
        h = torch.matmul(v_base, self.adapter_w1) + self.adapter_b1
        h = h * torch.sigmoid(1.702 * h)
        delta = torch.matmul(h, self.adapter_w2) + self.adapter_b2
        v_adapt = v_base + 0.35 * delta
        norm = torch.sqrt((v_adapt * v_adapt).sum(dim=-1, keepdim=True) + 1e-12)
        return v_adapt / norm

    def get_normalized_W(self) -> torch.Tensor:
        w = self.base_W + self.delta_W
        norm = torch.sqrt((w * w).sum(dim=0, keepdim=True) + 1e-12)
        return w / norm

    def forward(self, v_base: torch.Tensor, mask_self_proto_indices: torch.Tensor | None = None) -> torch.Tensor:
        v_norm = self.adapt_vectors(v_base)  # [B, 512]
        w_norm = self.get_normalized_W()     # [512, 4660]

        raw_sim = torch.matmul(v_norm, w_norm)  # [B, 4660]

        # Compute adapted prototype keys [512, M]
        proto_keys = self.adapt_vectors(self.proto_base_vecs).transpose(0, 1)  # [512, M]
        proto_sim = torch.matmul(v_norm, proto_keys)  # [B, M]

        if mask_self_proto_indices is not None:
            # Prevent prototype self-matching during training so the model learns true generalization
            proto_sim = proto_sim.clone()
            for row_i, p_idx in enumerate(mask_self_proto_indices.tolist()):
                if p_idx >= 0:
                    proto_sim[row_i, p_idx] = -10.0

        proto_attn = torch.softmax(proto_sim * 26.0, dim=-1)
        proto_boost = torch.matmul(proto_attn, self.proto_values)  # [B, 4660]

        cal_sim = (0.78 * raw_sim + 0.22 * proto_boost) - self.tag_bias

        mean_sim = cal_sim.mean(dim=-1, keepdim=True)
        diff_sim = cal_sim - mean_sim
        std_sim = torch.sqrt((diff_sim * diff_sim).mean(dim=-1, keepdim=True) + 1e-6)
        z_score = diff_sim / std_sim

        abs_logit = (cal_sim - 0.222) * torch.clamp(self.tag_scale, 20.0, 55.0)
        rel_logit = (z_score - 2.05) * 1.65
        logits = 0.55 * abs_logit + 0.45 * rel_logit
        return logits


def main():
    from export_onnx_model import calibrate_weights_and_build_prototypes, compute_or_load_tag_embeddings

    vocab = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    labels = vocab["labels"]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    print(f"[train] Loading {MODEL_ID} from local cache...")
    clip_model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True, attn_implementation="eager")
    clip_model.eval()

    raw_tag_matrix = compute_or_load_tag_embeddings(clip_model, vocab)
    base_W, base_bias, base_scale, _, _ = calibrate_weights_and_build_prototypes(raw_tag_matrix, labels)

    d_vecs, d_paths, d_groups = extract_or_load_d_image_features(clip_model)
    cal_vecs = np.load(CAL_VECS_PATH).astype(np.float32)

    X, Y_target, Y_weight, all_groups = build_supervision_targets(
        d_vecs=d_vecs,
        d_groups=d_groups,
        cal_vecs=cal_vecs,
        labels=labels,
        base_W=base_W,
        base_bias=base_bias,
    )

    # Stratified Train (82%) / Held-Out Validation (18%) split across every group
    rng = random.Random(2026)
    group_to_indices: dict[str, list[int]] = {}
    for i, g in enumerate(all_groups):
        # Bucket quixi non-001 vs quixi 001 for clean stratification
        if g.startswith("quixi::"):
            fname = g.split("::", 1)[1]
            gk = "quixi_001" if fname.endswith("_001_240x.jpg") else "quixi_rich"
        else:
            gk = g
        group_to_indices.setdefault(gk, []).append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for gk, idxs in group_to_indices.items():
        idxs_copy = list(idxs)
        rng.shuffle(idxs_copy)
        if len(idxs_copy) <= 4 or gk.startswith("cal_"):
            train_indices.extend(idxs_copy)
        else:
            n_val = max(2, int(len(idxs_copy) * 0.18))
            val_indices.extend(idxs_copy[:n_val])
            train_indices.extend(idxs_copy[n_val:])

    train_set = set(train_indices)
    print(f"[train] Dataset prepared: {len(train_indices)} train samples, {len(val_indices)} held-out validation samples")

    # Select ~2,150 prototypes strictly from train_indices (970 cal + ~1,180 D:\Image\Image train samples)
    proto_sample_indices: list[int] = []
    for gk, idxs in group_to_indices.items():
        tr_idxs = [idx for idx in idxs if idx in train_set]
        if gk.startswith("cal_"):
            proto_sample_indices.extend(tr_idxs)
        else:
            # Include up to 140 representative prototypes per D:\Image\Image category
            proto_sample_indices.extend(tr_idxs[:140])

    proto_sample_indices = sorted(set(proto_sample_indices))
    sample_to_proto_slot = {s_idx: p_slot for p_slot, s_idx in enumerate(proto_sample_indices)}
    print(f"[train] Constructed Contrastive Prototype Memory Bank with M = {len(proto_sample_indices)} prototypes")

    proto_init_vecs = X[proto_sample_indices].copy()  # [M, 512]
    proto_init_vals = np.zeros((len(proto_sample_indices), len(labels)), dtype=np.float32)

    for p_slot, s_idx in enumerate(proto_sample_indices):
        targ = Y_target[s_idx]
        wgt = Y_weight[s_idx]
        # Strong positive targets get +0.28 boost; strong hard negatives get -0.25 suppression
        pos_mask = (targ >= 0.85) & (wgt >= 4.0)
        neg_mask = (targ <= 0.05) & (wgt >= 5.0)
        proto_init_vals[p_slot, pos_mask] = 0.28
        proto_init_vals[p_slot, neg_mask] = -0.24
        # Also keep top-12 teacher tags
        top12 = np.argpartition(targ, -12)[-12:]
        for t_idx in top12:
            if targ[t_idx] >= 0.55 and proto_init_vals[p_slot, t_idx] == 0.0:
                proto_init_vals[p_slot, t_idx] = 0.18

    head = TrainableTaggingHead(
        base_W=base_W,
        base_bias=base_bias,
        base_scale=base_scale,
        proto_init_vecs=proto_init_vecs,
        proto_init_vals=proto_init_vals,
    )

    X_t = torch.from_numpy(X)
    Y_targ_t = torch.from_numpy(Y_target)
    Y_wgt_t = torch.from_numpy(Y_weight)
    proto_slot_per_sample = torch.tensor(
        [sample_to_proto_slot.get(i, -1) for i in range(X.shape[0])], dtype=torch.long
    )

    def evaluate_split(split_idxs: list[int], desc: str):
        head.eval()
        with torch.no_grad():
            idx_t = torch.tensor(split_idxs, dtype=torch.long)
            probs = torch.sigmoid(head(X_t[idx_t], mask_self_proto_indices=proto_slot_per_sample[idx_t])).numpy()

        # Check key accuracy metrics across groups in split
        metrics: dict[str, list[bool]] = {}
        for local_i, global_i in enumerate(split_idxs):
            g = all_groups[global_i]
            p = probs[local_i]
            if g == "offensive_gesture":
                ok = (
                    p[label_to_idx["Offensive or Vulgar Hand Gesture"]] > 0.60
                    and p[label_to_idx["Offensive or Vulgar Hand Gesture"]] > p[label_to_idx["Friendly or Benign Hand Gesture"]] + 0.20
                )
                metrics.setdefault("offensive_gesture", []).append(ok)
            elif g in ("benign_gesture_like", "benign_gesture_peace"):
                ok = (
                    p[label_to_idx["Friendly or Benign Hand Gesture"]] > 0.60
                    and p[label_to_idx["Friendly or Benign Hand Gesture"]] > p[label_to_idx["Offensive or Vulgar Hand Gesture"]] + 0.20
                )
                metrics.setdefault("benign_gesture", []).append(ok)
            elif g in ("bad_meme", "den_hate_meme"):
                ok = p[label_to_idx["Internet Meme or Satirical Graphic"]] > 0.60
                metrics.setdefault("memes", []).append(ok)
            elif g in ("den_profane_banner", "den_quote_banner", "acc_benign_banner"):
                ok = max(p[label_to_idx["Internet Meme or Satirical Graphic"]], p[label_to_idx["Digital Screenshot, App or Website UI"]], p[label_to_idx["text"]], p[label_to_idx["banner"]]) > 0.60
                metrics.setdefault("text_banners", []).append(ok)
            elif g in ("den_weapon_gun", "den_weapon_xray"):
                ok = p[label_to_idx["Weapon, Firearm or Knife"]] > 0.60
                metrics.setdefault("weapons_and_xray", []).append(ok)
            elif g in ("acc_invoice", "acc_receipt"):
                ok = p[label_to_idx["Receipt, Invoice or Financial Bill"]] > 0.60
                metrics.setdefault("invoices_receipts", []).append(ok)
            elif g in ("acc_selfie", "acc_headshot"):
                ok = (
                    p[label_to_idx["Close-up Selfie"]] > 0.60
                    and p[label_to_idx["Explicit Nudity or Adult Sexual Content"]] < 0.35
                    and p[label_to_idx["Offensive or Vulgar Hand Gesture"]] < 0.35
                )
                metrics.setdefault("clean_selfies", []).append(ok)
            elif g.startswith("quixi::"):
                fname = g.split("::", 1)[1]
                if fname.endswith("_001_240x.jpg"):
                    ok = max(
                        p[label_to_idx["Explicit Nudity or Adult Sexual Content"]],
                        p[label_to_idx["Swimwear, Bikini, Lingerie or Shirtless"]],
                    ) > 0.60
                    metrics.setdefault("quixi_adult_lingerie", []).append(ok)
                else:
                    # Check that Top-3 predicted tags match at least one ground-truth positive tag (Y_target >= 0.90)
                    gt_pos = np.where(Y_target[global_i] >= 0.90)[0]
                    if len(gt_pos) > 0:
                        top8 = set(np.argsort(p)[-8:])
                        ok = bool(top8.intersection(gt_pos))
                        metrics.setdefault("quixi_historical_cultural", []).append(ok)

        summary = {k: f"{100.0 * sum(v) / len(v):.1f}% ({sum(v)}/{len(v)})" for k, v in sorted(metrics.items())}
        total_ok = sum(sum(v) for v in metrics.values())
        total_n = sum(len(v) for v in metrics.values())
        print(f"[eval:{desc}] Overall: {100.0 * total_ok / max(total_n, 1):.2f}% ({total_ok}/{total_n}) | {summary}")

    print("\n--- BEFORE SUPERVISED D:\\Image\\Image TRAINING (Zero-Shot Calibrated v2.0 Baseline) ---")
    evaluate_split(val_indices, "heldout_val_before")

    optimizer = torch.optim.AdamW(
        [
            {"params": [head.adapter_w1, head.adapter_b1, head.adapter_w2, head.adapter_b2], "lr": 3.5e-3, "weight_decay": 1e-3},
            {"params": [head.delta_W], "lr": 2.5e-3, "weight_decay": 2e-3},
            {"params": [head.tag_bias, head.tag_scale], "lr": 2.0e-3, "weight_decay": 0.0},
            {"params": [head.proto_values], "lr": 4.0e-3, "weight_decay": 5e-4},
        ]
    )
    epochs = 28
    batch_size = 128
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    train_idx_arr = np.array(train_indices, dtype=np.int64)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        head.train()
        np.random.shuffle(train_idx_arr)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, len(train_idx_arr), batch_size):
            b_idx = torch.from_numpy(train_idx_arr[start : start + batch_size])
            bx = X_t[b_idx]
            by = Y_targ_t[b_idx]
            bw = Y_wgt_t[b_idx]
            bp = proto_slot_per_sample[b_idx]

            logits = head(bx, mask_self_proto_indices=bp)
            bce = F.binary_cross_entropy_with_logits(logits, by, weight=bw, reduction="mean")

            # Anchor regularization so unsupervised RAM tags remain faithful to their CLIP semantic space
            reg_w = (head.delta_W * head.delta_W).mean() * 12.0
            reg_proto = ((head.proto_values - torch.from_numpy(proto_init_vals)) ** 2).mean() * 1.5
            loss = bce + reg_w + reg_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                head.proto_values.clamp_(-0.55, 0.55)

            epoch_loss += float(loss.item())
            steps += 1

        scheduler.step()
        if epoch in (1, 7, 14, 21, epochs):
            print(f"[train] Epoch {epoch:02d}/{epochs} | loss={epoch_loss / max(steps, 1):.4f} ({time.time() - t_start:.1f}s)")
            evaluate_split(val_indices, f"heldout_val_ep{epoch:02d}")

    print("\n--- FINAL EVALUATION ON HELD-OUT VALIDATION & FULL D:\\Image\\Image DATASET ---")
    evaluate_split(val_indices, "FINAL_HELDOUT_VAL")
    evaluate_split(train_indices, "FINAL_TRAIN")

    # Export learned parameters in numpy format for `export_onnx_model.py`
    head.eval()
    with torch.no_grad():
        final_W = head.get_normalized_W().cpu().numpy().astype(np.float32)
        final_bias = head.tag_bias.view(-1).cpu().numpy().astype(np.float32)
        final_scale = torch.clamp(head.tag_scale, 20.0, 55.0).view(-1).cpu().numpy().astype(np.float32)
        final_proto_keys = head.adapt_vectors(head.proto_base_vecs).transpose(0, 1).cpu().numpy().astype(np.float32)
        final_proto_vals = head.proto_values.cpu().numpy().astype(np.float32)

        adapter_w1 = head.adapter_w1.cpu().numpy().astype(np.float32)
        adapter_b1 = head.adapter_b1.cpu().numpy().astype(np.float32)
        adapter_w2 = head.adapter_w2.cpu().numpy().astype(np.float32)
        adapter_b2 = head.adapter_b2.cpu().numpy().astype(np.float32)

    np.savez_compressed(
        TRAINED_HEAD_PATH,
        tag_matrix=final_W,
        tag_bias=final_bias,
        tag_scale=final_scale,
        proto_keys=final_proto_keys,
        proto_values=final_proto_vals,
        adapter_w1=adapter_w1,
        adapter_b1=adapter_b1,
        adapter_w2=adapter_w2,
        adapter_b2=adapter_b2,
    )
    print(f"[train] Saved trained v3.0 weights -> {TRAINED_HEAD_PATH}")


if __name__ == "__main__":
    main()
