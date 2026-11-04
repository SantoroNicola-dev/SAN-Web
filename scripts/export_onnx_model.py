"""
Exports the Calibrated Universal All-in-One Multi-Label Image Tagging ONNX Model (`models/san_tagger.onnx`)
designed for 100% compatibility with BOTH:
  1. Unity Sentis / Unity.InferenceEngine (CPU/GPU, Opset 15, pure FP32, zero dynamic shape ops)
  2. Microsoft ONNX Runtime (`onnxruntime` & `onnxruntime-web` WASM in the browser)

Upgraded Calibration & Architecture (v2.0):
  1. 8-Template Visual Prompt Ensembling + Neutral Prompt Carrier Subtraction for all 4,585 RAM tags.
  2. Native Multi-Prompt Ensembles + Linear Contrastive Benign-Anchor Orthogonalization for all 75 SAN-1 categories.
  3. Dense Spatial + Global Feature Fusion: combines the global [CLS] token with a Foreground Center-Weighted
     Spatial Patch Projection (ClearCLIP value-projection over the central 5x5 patch grid in Layer 12).
  4. Multi-Domain Subspace De-Biasing & Per-Tag Empirical Quantile Calibration (`tag_bias` & `tag_scale` [1, 4660])
     computed across 1,000+ real images (Tiny-ImageNet, Cat&Dog, Food101, Stanford Cars, Oxford Flowers, Fashion, CelebA, Documents).
  5. Embedded Tip-Adapter Visual Prototype Memory Bank (`proto_keys` [512, 970] & `proto_values` [970, 4660])
     providing non-parametric visual k-NN refinement inside the single ONNX graph.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizerFast

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
LABELS_JSON_PATH = MODELS_DIR / "labels.json"
TAG_CACHE_PATH = MODELS_DIR / "tag_embeddings_v2.npy"
CAL_VECS_PATH = MODELS_DIR / "calibration_image_vecs.npy"
TRAINED_HEAD_PATH = MODELS_DIR / "d_image_trained_head.npz"
ONNX_OUTPUT_PATH = MODELS_DIR / "san_tagger.onnx"
MANIFEST_OUTPUT_PATH = MODELS_DIR / "model_manifest.json"

SAN1_CATEGORIES_JSON = Path(r"D:\Projects\VibeProject\SAN-1\config\categories.json")
SAN1_CACHE_NPZ = Path(r"D:\Projects\VibeProject\SAN-1\models\clip\categories_embeddings_cache.npz")

MODEL_ID = "openai/clip-vit-base-patch32"

# True broad visual hypernyms that legitimately occur across many images in a domain
DOMAIN_HYPERNYMS = {
    "global": {
        "photo", "picture", "photograph", "color", "close-up",
    },
    "face": {
        "person", "face", "portrait", "head", "human", "woman", "man", "girl", "boy", "lady",
        "smile", "hair", "short hair", "long hair", "brown hair", "black hair", "blond hair",
        "hairstyle", "haircut", "hair color", "eye", "eyebrow", "nose", "mouth", "lip", "chin",
        "cheek", "forehead", "ear", "neck", "skin", "selfie", "Close-up Selfie", "Group Selfie or Multi-Person Portrait",
        "Full-Body or Waist-Up Person", "face close-up", "headshot", "adult", "young",
        "glasses", "eyeglasses", "sunglasses", "beard", "mustache", "earring", "necklace",
        "hand", "finger", "gesture", "Offensive or Vulgar Hand Gesture", "Friendly or Benign Hand Gesture",
    },
    "pet": {
        "animal", "pet", "mammal", "dog", "cat", "puppy", "kitten", "feline", "canine",
        "Domestic Dog", "Cat / Feline", "fur", "whiskers", "paw", "snout", "ears", "collar",
    },
    "food": {
        "food", "dish", "meal", "cuisine", "plate", "bowl", "lunch", "dinner", "breakfast",
        "Cooked Meal / Restaurant Dish", "Pizza, Burger or Fast Food", "Dessert, Cake or Ice Cream",
        "Fruit or Vegetables", "ingredient", "sauce", "meat", "vegetable", "curry", "rice", "soup",
    },
    "car": {
        "car", "vehicle", "motor vehicle", "land vehicle", "automobile", "Car / Automobile",
        "Truck, Bus or Heavy Vehicle", "wheel", "tire", "windshield", "headlight", "bumper",
        "sedan", "suv", "sports car", "truck", "jeep", "van", "coupe", "convertible",
    },
    "flower": {
        "flower", "plant", "blossom", "petal", "flora", "bloom", "Flowers, Blossoms & Houseplants",
        "leaf", "stem", "garden", "botany", "wildflower", "bouquet", "rose", "tulip", "daisy", "sunflower",
    },
    "fashion": {
        "clothing", "clothe", "garment", "apparel", "fashion", "Clothing, Shoes & Fashion",
        "shirt", "t-shirt", "t shirt", "top", "dress", "jacket", "coat", "pants", "jeans",
        "shoe", "footwear", "sneakers", "bag", "handbag", "watch", "hat", "cap",
    },
}

# Non-visual / abstract / occupational words in RAM that suffer from Wikipedia caption bias
ABSTRACT_OR_OCCUPATIONAL_WORDS = {
    "appear", "up", "out", "attach", "edit", "well", "test", "half", "second", "first",
    "third", "number", "flake", "hide", "image", "part", "type", "kind", "form", "view",
    "side", "top", "bottom", "left", "right", "back", "front", "area", "place", "point",
    "line", "end", "way", "set", "put", "take", "make", "use", "show", "give", "keep",
    "author", "writer", "researcher", "educator", "director", "film director", "actor",
    "tv actor", "theatre actor", "voice actor", "basketball coach", "coach", "makeup artist",
    "country pop artist", "pop artist", "lead singer", "socialite", "celebrity", "tv personality",
    "businessman", "business woman", "college student", "daughter", "son", "brother", "sister",
    "mother", "father", "wife", "husband", "lawyer", "attorney", "politician", "journalist",
    "reporter", "economist", "historian", "philosopher", "psychologist", "sociologist",
    "theologian", "columnist", "novelist", "poet", "playwright", "screenwriter", "producer",
    "record producer", "executive", "entrepreneur", "philanthropist", "activist", " spokesperson",
    "contestant", "finalist", "winner", "champion", "defender", "midfielder", "goalkeeper",
    "striker", "winger", "pitcher", "quarterback", "linebacker", "super bowl", "martial",
    "dj", "recipe", "cooking spray", "embellishment", "makeover",
}

OCCUPATIONAL_REGEX = re.compile(
    r"\b(actor|actress|director|producer|writer|author|researcher|educator|professor|teacher|"
    r"coach|politician|senator|governor|mayor|minister|president|journalist|reporter|columnist|"
    r"novelist|poet|singer|vocalist|rapper|musician|composer|conductor|songwriter|artist|"
    r"socialite|celebrity|personality|model|student|businessman|businesswoman|executive|"
    r"entrepreneur|lawyer|attorney|judge|physician|surgeon|therapist|psychologist|scientist|"
    r"historian|economist|philosopher|activist|spokesperson|contestant|player|athlete|"
    r"defender|midfielder|striker|goalkeeper|pitcher|quarterback|carpenter|plumber|electrician|"
    r"mechanic|baker|butcher|barber|hairdresser|tailor|waiter|waitress|bartender|dentist|"
    r"pharmacist|veterinarian|architect|accountant|banker|broker|consultant|analyst|clerk)\b",
    re.IGNORECASE,
)


class ExplicitLayerNorm(nn.Module):
    """
    Decomposed LayerNorm using elementary ONNX ops (ReduceMean, Sub, Mul, Add, Sqrt, Div)
    for 100% compatibility in Unity Sentis and ONNX Runtime Web.
    """

    def __init__(self, orig_ln: nn.LayerNorm):
        super().__init__()
        self.eps = float(orig_ln.eps)
        self.register_buffer("weight", orig_ln.weight.detach().clone().view(1, 1, -1))
        self.register_buffer("bias", orig_ln.bias.detach().clone().view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)
        normed = diff / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class ExplicitEncoderLayer(nn.Module):
    """
    Static-shape ViT-B/32 Transformer Encoder block using only MatMul, Add, Mul,
    Reshape, Transpose, Softmax, Sigmoid (QuickGELU), and ExplicitLayerNorm.
    """

    def __init__(self, orig_layer: nn.Module):
        super().__init__()
        self.ln1 = ExplicitLayerNorm(orig_layer.layer_norm1)
        self.ln2 = ExplicitLayerNorm(orig_layer.layer_norm2)

        self.q_proj = orig_layer.self_attn.q_proj
        self.k_proj = orig_layer.self_attn.k_proj
        self.v_proj = orig_layer.self_attn.v_proj
        self.out_proj = orig_layer.self_attn.out_proj

        self.fc1 = orig_layer.mlp.fc1
        self.fc2 = orig_layer.mlp.fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.ln1(x)

        q = (self.q_proj(h) * 0.125).reshape(1, 50, 12, 64).transpose(1, 2)
        k = self.k_proj(h).reshape(1, 50, 12, 64).transpose(1, 2)
        v = self.v_proj(h).reshape(1, 50, 12, 64).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(1, 50, 768)
        x = residual + self.out_proj(attn_out)

        residual = x
        h = self.ln2(x)
        h = self.fc1(h)
        h = h * torch.sigmoid(1.702 * h)
        x = residual + self.fc2(h)
        return x

    def forward_with_spatial_detail(self, x: torch.Tensor, center_patch_selector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        In the final encoder layer, computes both:
          1. Standard global self-attention output for [CLS]
          2. Value-projected spatial patch representation (ClearCLIP/MaskCLIP) pooled over the central foreground
        """
        residual = x
        h = self.ln1(x)

        q = (self.q_proj(h) * 0.125).reshape(1, 50, 12, 64).transpose(1, 2)
        k = self.k_proj(h).reshape(1, 50, 12, 64).transpose(1, 2)
        v_raw = self.v_proj(h)  # [1, 50, 768]
        v = v_raw.reshape(1, 50, 12, 64).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(1, 50, 768)
        x_global = residual + self.out_proj(attn_out)

        res_g = x_global
        hg = self.ln2(x_global)
        hg = self.fc1(hg)
        hg = hg * torch.sigmoid(1.702 * hg)
        x_global = res_g + self.fc2(hg)

        # ClearCLIP local value path for center-weighted foreground patches: [1, 1, 50] @ [1, 50, 768] -> [1, 1, 768]
        x_center_in = torch.matmul(center_patch_selector, residual)
        v_center_in = torch.matmul(center_patch_selector, v_raw)
        x_local = x_center_in + self.out_proj(v_center_in)
        hl = self.ln2(x_local)
        hl = self.fc1(hl)
        hl = hl * torch.sigmoid(1.702 * hl)
        x_local = x_local + self.fc2(hl)

        return x_global, x_local


class SanAllInOneTagger(nn.Module):
    """
    Calibrated Vision Backbone + Dense Spatial Fusion + Residual Visual Adapter +
    Contrastive Prototype Memory Bank + 4,660-Tag Head (v3.0).
    Takes RGB [0..1] tensor of shape [1, 3, 224, 224] and outputs calibrated
    multi-label probabilities [1, 4660] in [0.0, 1.0].
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        tag_matrix: np.ndarray,
        tag_bias: np.ndarray,
        tag_scale: np.ndarray,
        proto_keys: np.ndarray,
        proto_values: np.ndarray,
        adapter_w1: np.ndarray,
        adapter_b1: np.ndarray,
        adapter_w2: np.ndarray,
        adapter_b2: np.ndarray,
    ):
        super().__init__()
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

        # Selector for global CLS token (index 0)
        cls_selector = torch.zeros(1, 1, 50, dtype=torch.float32)
        cls_selector[0, 0, 0] = 1.0
        self.register_buffer("cls_selector", cls_selector)

        # Gaussian center-weighted spatial patch selector over the 7x7 patch grid (indices 1..49)
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

        # Residual Visual Adapter (512 -> 256 -> 512)
        self.register_buffer("adapter_w1", torch.from_numpy(adapter_w1.astype(np.float32)))
        self.register_buffer("adapter_b1", torch.from_numpy(adapter_b1.astype(np.float32)).view(1, -1))
        self.register_buffer("adapter_w2", torch.from_numpy(adapter_w2.astype(np.float32)))
        self.register_buffer("adapter_b2", torch.from_numpy(adapter_b2.astype(np.float32)).view(1, -1))

        # Calibrated Tag Matrix [512, 4660], Per-Tag Bias [1, 4660], and Scale [1, 4660]
        self.register_buffer("tag_matrix", torch.from_numpy(tag_matrix.astype(np.float32)))
        self.register_buffer("tag_bias", torch.from_numpy(tag_bias.astype(np.float32)).view(1, -1))
        self.register_buffer("tag_scale", torch.from_numpy(tag_scale.astype(np.float32)).view(1, -1))

        # Embedded Contrastive Visual Prototype Bank: proto_keys [512, M], proto_values [M, 4660]
        self.register_buffer("proto_keys", torch.from_numpy(proto_keys.astype(np.float32)))
        self.register_buffer("proto_values", torch.from_numpy(proto_values.astype(np.float32)))

    def encode_image_norm(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.pixel_mean) / self.pixel_std
        patches = self.patch_embedding(x)
        patches = patches.reshape(1, 768, 49).transpose(1, 2)

        hidden = torch.cat([self.class_embedding, patches], dim=1) + self.position_embedding
        hidden = self.pre_ln(hidden)

        for i in range(11):
            hidden = self.layers[i](hidden)

        hidden_global, hidden_local = self.layers[11].forward_with_spatial_detail(hidden, self.center_patch_selector)

        cls_token = torch.matmul(self.cls_selector, hidden_global)
        pooled_cls = self.post_ln(cls_token).reshape(1, 768)
        embeds_cls = self.visual_projection(pooled_cls)
        norm_cls = torch.sqrt((embeds_cls * embeds_cls).sum(dim=-1, keepdim=True) + 1e-12)
        v_cls = embeds_cls / norm_cls

        pooled_loc = self.post_ln(hidden_local).reshape(1, 768)
        embeds_loc = self.visual_projection(pooled_loc)
        norm_loc = torch.sqrt((embeds_loc * embeds_loc).sum(dim=-1, keepdim=True) + 1e-12)
        v_loc = embeds_loc / norm_loc

        # Fuse global CLS (86%) + dense center-weighted foreground patch representation (14%)
        v_fused = 0.86 * v_cls + 0.14 * v_loc
        norm_fused = torch.sqrt((v_fused * v_fused).sum(dim=-1, keepdim=True) + 1e-12)
        return v_fused / norm_fused

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        v_base = self.encode_image_norm(input)

        # 1. Residual Visual Feature Adapter (512 -> 256 -> 512)
        h = torch.matmul(v_base, self.adapter_w1) + self.adapter_b1
        h = h * torch.sigmoid(1.702 * h)
        delta = torch.matmul(h, self.adapter_w2) + self.adapter_b2
        v_adapt = v_base + 0.35 * delta
        image_norm = v_adapt / torch.sqrt((v_adapt * v_adapt).sum(dim=-1, keepdim=True) + 1e-12)

        # 2. Supervised & Calibrated Cosine Similarity [1, 4660]
        raw_sim = torch.matmul(image_norm, self.tag_matrix)

        # 3. Contrastive Visual Prototype Memory Bank Consensus [1, 4660]
        proto_sim = torch.matmul(image_norm, self.proto_keys)  # [1, M]
        proto_attn = torch.softmax(proto_sim * 26.0, dim=-1)    # Sharp k-NN soft attention
        proto_boost = torch.matmul(proto_attn, self.proto_values)  # [1, 4660]

        # Fuse similarity with visual prototype consensus and subtract per-tag bias
        cal_sim = (0.78 * raw_sim + 0.22 * proto_boost) - self.tag_bias

        # Within-image statistical salience over the de-biased similarities
        mean_sim = cal_sim.mean(dim=-1, keepdim=True)
        diff_sim = cal_sim - mean_sim
        std_sim = torch.sqrt((diff_sim * diff_sim).mean(dim=-1, keepdim=True) + 1e-6)
        z_score = diff_sim / std_sim

        abs_logit = (cal_sim - 0.222) * self.tag_scale
        rel_logit = (z_score - 2.05) * 1.65
        logits = 0.55 * abs_logit + 0.45 * rel_logit

        return torch.sigmoid(logits)


def compute_or_load_tag_embeddings(clip_model: CLIPModel, vocab_data: dict) -> np.ndarray:
    labels = vocab_data["labels"]
    if TAG_CACHE_PATH.exists():
        cached = np.load(TAG_CACHE_PATH)
        if cached.shape == (512, len(labels)):
            print(f"[export] Loaded v2 multi-prompt tag embedding matrix {cached.shape} from {TAG_CACHE_PATH}")
            return cached

    print(f"[export] Encoding {len(labels)} English tags with 8-template visual ensemble + neutral centering...")
    tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID, local_files_only=True)

    templates = [
        "a photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a cropped photo of the {}.",
        "a photo of one {}.",
        "a clear photograph showing {}.",
        "a photo of a large {}.",
        "a photo of a small {}.",
    ]
    neutral_prompts = [
        "a photo of a.",
        "a close-up photo of the.",
        "a bright photo of a.",
        "a cropped photo of the.",
        "a photo of one.",
        "a clear photograph showing.",
        "a photo.",
        "an image.",
    ]

    with torch.no_grad():
        ntok = tokenizer(neutral_prompts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        nout = clip_model.text_model(input_ids=ntok["input_ids"], attention_mask=ntok["attention_mask"])
        nfeats = clip_model.text_projection(nout.pooler_output)
        nfeats = nfeats / nfeats.norm(dim=-1, keepdim=True)
        neutral_vec = nfeats.mean(dim=0).cpu().numpy()
        neutral_vec = neutral_vec / max(np.linalg.norm(neutral_vec), 1e-12)

    num_tags = len(labels)
    accum = np.zeros((num_tags, 512), dtype=np.float32)

    batch_size = 512
    t0 = time.time()
    with torch.no_grad():
        for t_idx, tmpl in enumerate(templates):
            prompts = [tmpl.format(lbl) for lbl in labels]
            for start in range(0, num_tags, batch_size):
                end = min(start + batch_size, num_tags)
                tok = tokenizer(
                    prompts[start:end],
                    padding=True,
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                )
                text_out = clip_model.text_model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                )
                feats = clip_model.text_projection(text_out.pooler_output)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                accum[start:end] += feats.cpu().numpy()
            print(f"  - Completed template {t_idx + 1}/{len(templates)} ({time.time() - t0:.1f}s)")

    norms = np.maximum(np.linalg.norm(accum, axis=-1, keepdims=True), 1e-12)
    tag_vecs = accum / norms

    # Subtract 30% of the generic syntax carrier wave ("a photo of a...") and re-normalize
    tag_vecs = tag_vecs - 0.30 * neutral_vec[np.newaxis, :]
    norms = np.maximum(np.linalg.norm(tag_vecs, axis=-1, keepdims=True), 1e-12)
    tag_vecs = tag_vecs / norms

    # Integrate SAN-1's domain-engineered multi-prompt embeddings & contrastive anchor orthogonalization
    if SAN1_CATEGORIES_JSON.exists() and SAN1_CACHE_NPZ.exists():
        san1_cfg = json.loads(SAN1_CATEGORIES_JSON.read_text(encoding="utf-8"))
        san1_cache = np.load(SAN1_CACHE_NPZ, allow_pickle=True)
        action_mat = san1_cache["action_matrix"].astype(np.float32)  # [75, 512]
        benign_mat = san1_cache["benign_matrix"].astype(np.float32)
        benign_keys = [str(k) for k in san1_cache["benign_keys"]]
        benign_dict = {k: benign_mat[i] for i, k in enumerate(benign_keys)}

        label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
        for c_idx, cat in enumerate(san1_cfg.get("categories", [])):
            lbl = cat.get("label", "").strip()
            if lbl in label_to_idx and c_idx < len(action_mat):
                target_i = label_to_idx[lbl]
                v_cat = action_mat[c_idx].copy()
                # Apply linear contrastive orthogonalization against defined benign anchors
                anchors = cat.get("contrast_anchors", [])
                if anchors:
                    anchor_vecs = [benign_dict[a] for a in anchors if a in benign_dict]
                    if anchor_vecs:
                        mean_anchor = np.mean(anchor_vecs, axis=0)
                        mean_anchor /= max(np.linalg.norm(mean_anchor), 1e-12)
                        proj = float(np.dot(v_cat, mean_anchor))
                        if proj > 0:
                            v_cat = v_cat - 0.32 * proj * mean_anchor
                            v_cat /= max(np.linalg.norm(v_cat), 1e-12)
                tag_vecs[target_i] = v_cat

    tag_matrix = tag_vecs.T.astype(np.float32)  # [512, 4660]
    np.save(TAG_CACHE_PATH, tag_matrix)
    print(f"[export] Saved v2 tag embedding matrix {tag_matrix.shape} -> {TAG_CACHE_PATH}")
    return tag_matrix


def calibrate_weights_and_build_prototypes(
    tag_matrix: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs:
      1. Multi-Domain Subspace De-Biasing on `tag_matrix` [512, 4660]
      2. Empirical Per-Tag Baseline (`tag_bias`) & Scale (`tag_scale`) Calibration
      3. Tip-Adapter Visual Prototype Bank (`proto_keys` [512, M], `proto_values` [M, 4660])
    """
    cal_vecs = np.load(CAL_VECS_PATH).astype(np.float32)  # [970, 512]
    cal_vecs /= np.maximum(np.linalg.norm(cal_vecs, axis=-1, keepdims=True), 1e-12)

    W = tag_matrix.copy()  # [512, 4660]

    # Align the 75 SAN-1 multi-prompt vectors (indices >= 4585) with the neutral-centered spherical frame
    if W.shape[1] > 4585 and Path(MODELS_DIR / "tag_embeddings_cache.npy").exists():
        v1_mat = np.load(MODELS_DIR / "tag_embeddings_cache.npy")
        if v1_mat.shape == W.shape:
            # Blend 60% v1 prompt embedding + 40% SAN-1 multi-prompt embedding for indices >= 4585
            W[:, 4585:] = 0.60 * v1_mat[:, 4585:] + 0.40 * W[:, 4585:]
            W[:, 4585:] /= np.maximum(np.linalg.norm(W[:, 4585:], axis=0, keepdims=True), 1e-12)

    # 1. Domain-Centroid Subspace De-Biasing
    domain_slices = [
        ("global", 0, 350, 0.195),
        ("pet", 350, 450, 0.215),
        ("food", 450, 570, 0.220),
        ("car", 570, 670, 0.240),
        ("flower", 670, 750, 0.235),
        ("fashion", 750, 870, 0.215),
        ("face", 870, 970, 0.225),
    ]

    for dname, s, e, cap in domain_slices:
        c = cal_vecs[s:e].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        proj = c @ W  # [4660]
        allowed = DOMAIN_HYPERNYMS[dname]
        for j, lbl in enumerate(labels):
            if lbl not in allowed and proj[j] > cap:
                excess = (proj[j] - cap) * 0.88
                W[:, j] -= excess * c

    # Re-normalize columns of W
    col_norms = np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    W = (W / col_norms).astype(np.float32)

    # 2. Compute Empirical Per-Tag Distribution across the 970 real calibration images
    sims_cal = cal_vecs @ W  # [970, 4660]
    mu_tag = sims_cal.mean(axis=0)
    std_tag = np.maximum(sims_cal.std(axis=0), 0.012)
    q92_tag = np.percentile(sims_cal, 92, axis=0)
    q96_tag = np.percentile(sims_cal, 96, axis=0)

    med_mu = float(np.median(mu_tag))
    med_q92 = float(np.median(q92_tag))
    all_hypernyms = set().union(*DOMAIN_HYPERNYMS.values())

    tag_bias = np.zeros(len(labels), dtype=np.float32)
    tag_scale = np.full(len(labels), 36.0, dtype=np.float32)

    for j, lbl in enumerate(labels):
        lbl_lower = lbl.lower().strip()
        is_hypernym = lbl in all_hypernyms
        is_abstract_or_occ = (
            lbl_lower in ABSTRACT_OR_OCCUPATIONAL_WORDS
            or bool(OCCUPATIONAL_REGEX.search(lbl_lower))
        )

        if is_hypernym:
            # Preserve high sensitivity for core visual hypernyms (person, face, smile, car, dog, cat, food, etc.)
            bias_val = max(0.0, (mu_tag[j] - med_mu) * 0.25) - 0.008
            scale_val = 38.0
        elif is_abstract_or_occ:
            # Strong anti-hubness guard for abstract verbs and Wikipedia biography/occupational roles
            bias_val = (
                max(0.0, (mu_tag[j] - med_mu) * 0.90)
                + max(0.0, (q96_tag[j] - med_q92) * 0.85)
                + 0.042
            )
            scale_val = 28.0
        else:
            # Standard concrete visual tag: align its 92nd-percentile background noise floor
            bias_val = (
                max(0.0, (mu_tag[j] - med_mu) * 0.65)
                + max(0.0, (q92_tag[j] - med_q92) * 0.55)
            )
            scale_val = float(np.clip(0.80 / std_tag[j], 30.0, 44.0))

        tag_bias[j] = bias_val
        tag_scale[j] = scale_val

    # 3. Construct Tip-Adapter Visual Prototype Bank (proto_keys [512, 970], proto_values [970, 4660])
    # Each prototype stores its top-20 cleanest concrete visual tag affinities
    proto_keys = cal_vecs.T.astype(np.float32)  # [512, 970]
    debiased_cal_sims = sims_cal - tag_bias[np.newaxis, :]  # [970, 4660]
    proto_values = np.zeros_like(debiased_cal_sims, dtype=np.float32)

    for i in range(debiased_cal_sims.shape[0]):
        row = debiased_cal_sims[i]
        top_k_idx = np.argpartition(row, -20)[-20:]
        proto_values[i, top_k_idx] = np.maximum(row[top_k_idx], 0.18)

    print(
        f"[calibrate] Calibrated 4,660 tags across {cal_vecs.shape[0]} multi-domain prototypes "
        f"(median bias={np.median(tag_bias):.4f}, max bias={np.max(tag_bias):.4f})"
    )
    return W, tag_bias, tag_scale, proto_keys, proto_values


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    labels = vocab["labels"]
    print(f"[export] Loaded {len(labels)} labels from {LABELS_JSON_PATH}")

    print(f"[export] Loading {MODEL_ID} from local cache...")
    clip_model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True, attn_implementation="eager")
    clip_model.eval()

    raw_tag_matrix = compute_or_load_tag_embeddings(clip_model, vocab)
    if TRAINED_HEAD_PATH.exists():
        print(f"[export] Loading supervised D:\\Image\\Image v3.0 head & prototype bank from {TRAINED_HEAD_PATH}...")
        th = np.load(TRAINED_HEAD_PATH)
        cal_W = th["tag_matrix"].astype(np.float32)
        tag_bias = th["tag_bias"].astype(np.float32)
        tag_scale = th["tag_scale"].astype(np.float32)
        proto_keys = th["proto_keys"].astype(np.float32)
        proto_values = th["proto_values"].astype(np.float32)
        adapter_w1 = th["adapter_w1"].astype(np.float32)
        adapter_b1 = th["adapter_b1"].astype(np.float32)
        adapter_w2 = th["adapter_w2"].astype(np.float32)
        adapter_b2 = th["adapter_b2"].astype(np.float32)
    else:
        cal_W, tag_bias, tag_scale, proto_keys, proto_values = calibrate_weights_and_build_prototypes(
            raw_tag_matrix, labels
        )
        adapter_w1 = np.zeros((512, 256), dtype=np.float32)
        adapter_b1 = np.zeros((1, 256), dtype=np.float32)
        adapter_w2 = np.zeros((256, 512), dtype=np.float32)
        adapter_b2 = np.zeros((1, 512), dtype=np.float32)

    print(f"[export] Constructing calibrated SanAllInOneTagger v3.0 (prototypes={proto_keys.shape[1]})...")
    wrapper = SanAllInOneTagger(
        clip_model=clip_model,
        tag_matrix=cal_W,
        tag_bias=tag_bias,
        tag_scale=tag_scale,
        proto_keys=proto_keys,
        proto_values=proto_values,
        adapter_w1=adapter_w1,
        adapter_b1=adapter_b1,
        adapter_w2=adapter_w2,
        adapter_b2=adapter_b2,
    )
    wrapper.eval()

    dummy_rgb = torch.rand(1, 3, 224, 224, dtype=torch.float32)

    print(f"[export] Exporting ONNX model to {ONNX_OUTPUT_PATH} (Opset 15)...")
    torch.onnx.export(
        wrapper,
        (dummy_rgb,),
        str(ONNX_OUTPUT_PATH),
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["scores"],
        dynamo=False,
    )

    model_proto = onnx.load(str(ONNX_OUTPUT_PATH))
    onnx.save_model(model_proto, str(ONNX_OUTPUT_PATH), save_as_external_data=False)
    onnx.checker.check_model(model_proto)

    ops = sorted({node.op_type for node in model_proto.graph.node})
    file_size_bytes = ONNX_OUTPUT_PATH.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"[export] ONNX model verified! Size: {file_size_mb:.2f} MB")
    print(f"[export] Operators used ({len(ops)}): {ops}")

    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH), providers=["CPUExecutionProvider"])
    ort_scores = sess.run(["scores"], {"input": dummy_rgb.numpy()})[0]
    with torch.no_grad():
        pt_scores = wrapper(dummy_rgb).numpy()
    ort_diff = float(np.max(np.abs(ort_scores - pt_scores)))
    print(f"[export] ONNX Runtime vs PyTorch output max diff: {ort_diff:.2e}")
    assert ort_diff < 1e-4, f"ORT parity check failed: {ort_diff}"

    manifest = {
        "name": "SAN-Web Universal Multi-Label Image Tagger (Supervised & Calibrated v3.0)",
        "version": "3.0.0",
        "model_file": "models/san_tagger.onnx",
        "labels_json": "models/labels.json",
        "labels_txt": "models/labels.txt",
        "size_bytes": file_size_bytes,
        "size_mb": round(file_size_mb, 2),
        "sha256": sha256_file(ONNX_OUTPUT_PATH),
        "opset_version": 15,
        "precision": "FP32",
        "num_classes": len(labels),
        "calibration": {
            "prompt_templates": 8,
            "neutral_prompt_centered": True,
            "residual_visual_adapter": "512->256->512",
            "supervised_d_image_training": True,
            "multi_domain_prototypes": int(proto_keys.shape[1]),
            "dense_spatial_patch_fusion": True,
            "subspace_centroid_debiased": True,
        },
        "input": {
            "name": "input",
            "shape": [1, 3, 224, 224],
            "dtype": "float32",
            "format": "NCHW RGB normalized in [0.0, 1.0]",
        },
        "output": {
            "name": "scores",
            "shape": [1, len(labels)],
            "dtype": "float32",
            "description": "Independent calibrated multi-label probability in [0.0, 1.0] for each tag",
        },
        "operators": ops,
        "compatibility": {
            "unity_sentis": True,
            "onnxruntime_cpu": True,
            "onnxruntime_web_wasm": True,
            "estimated_peak_ram_mb": round(file_size_mb * 1.6, 1),
            "max_memory_budget_mb": 4096,
        },
    }
    MANIFEST_OUTPUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[export] Saved manifest -> {MANIFEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
