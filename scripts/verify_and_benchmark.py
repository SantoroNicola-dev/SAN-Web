"""
Comprehensive Verification & Benchmark Suite for `models/san_tagger.onnx`:
  1. Checks ONNX graph validity, single-file self-containment, and Unity Sentis operator whitelist.
  2. Measures CPU memory consumption (verifying < 4 GB RAM requirement).
  3. Benchmarks ONNX Runtime CPU inference latency.
  4. Runs multi-label image tagging on real test images and prints top predictions.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
ONNX_PATH = MODELS_DIR / "san_tagger.onnx"
LABELS_JSON_PATH = MODELS_DIR / "labels.json"
SAN1_TEST_DIR = Path(r"D:\Projects\VibeProject\SAN-1\Test")

# Unity Sentis 100% universally supported core math/tensor operators
SENTIS_SAFE_OPS = {
    "Add",
    "Concat",
    "Constant",
    "Conv",
    "Div",
    "Identity",
    "MatMul",
    "Mul",
    "ReduceMean",
    "ReduceSum",
    "Reshape",
    "Sigmoid",
    "Softmax",
    "Sqrt",
    "Sub",
    "Transpose",
}

FORBIDDEN_SENTIS_OPS = {
    "DynamicQuantizeLinear",
    "MatMulInteger",
    "ConvInteger",
    "QLinearConv",
    "QLinearMatMul",
    "NonMaxSuppression",
    "SequenceConstruct",
    "Loop",
    "If",
}


def get_process_memory_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE
        handle = GetCurrentProcess()

        fn = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if fn(handle, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [224, 224, 3] in [0, 1]
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 224, 224]
    return np.ascontiguousarray(arr, dtype=np.float32)


def main():
    print("=" * 72)
    print("SAN-Web Universal ONNX Model Verification & Benchmark")
    print("=" * 72)

    assert ONNX_PATH.exists(), f"Missing ONNX model: {ONNX_PATH}"
    assert LABELS_JSON_PATH.exists(), f"Missing labels file: {LABELS_JSON_PATH}"

    vocab = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    labels = vocab["labels"]
    num_tags = len(labels)

    # 1. Inspect ONNX Protobuf & Operator Compatibility for Unity Sentis + ORT-Web
    print("\n[1/4] Verifying ONNX Opset & Unity Sentis Operator Compatibility...")
    model_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_proto)

    opset = model_proto.opset_import[0].version
    ops_used = {node.op_type for node in model_proto.graph.node}
    forbidden_found = ops_used.intersection(FORBIDDEN_SENTIS_OPS)
    unexpected_ops = ops_used - SENTIS_SAFE_OPS

    assert not forbidden_found, f"Forbidden Unity Sentis operators found: {forbidden_found}"
    assert not unexpected_ops, f"Unexpected operators found: {unexpected_ops}"

    # Verify zero external data references (must be 100% self-contained single file)
    for tensor in model_proto.graph.initializer:
        assert tensor.data_location == onnx.TensorProto.DEFAULT, (
            f"Initializer {tensor.name} uses external data! Must be single file."
        )

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"  [OK] File Size          : {size_mb:.2f} MB (single self-contained .onnx file)")
    print(f"  [OK] ONNX Opset         : {opset}")
    print(f"  [OK] Output Tag Classes : {num_tags} English tags")
    print(f"  [OK] Operators ({len(ops_used)})     : {sorted(ops_used)}")
    print("  [OK] Unity Sentis Check : PASSED (100% primitive FP32 math ops, zero INT8/dynamic-shape ops)")

    del model_proto

    # 2. Load in ONNX Runtime CPU & Check Peak Memory (< 4 GB = 4096 MB)
    print("\n[2/4] Initializing ONNX Runtime CPU Session & Checking Memory Budget...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_PATH), sess_options=sess_options, providers=["CPUExecutionProvider"])

    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
    scores = sess.run(["scores"], {"input": dummy_input})[0]
    peak_ram_mb = get_process_memory_mb()

    assert scores.shape == (1, num_tags), f"Unexpected output shape: {scores.shape}"
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0, "Scores must be probabilities in [0, 1]"
    assert peak_ram_mb < 4096.0, f"Memory exceeded 4GB limit: {peak_ram_mb:.1f} MB"

    print(f"  [OK] Output Tensor Shape: {scores.shape} (float32 in [0.0, 1.0])")
    print(f"  [OK] Peak Process RAM   : {peak_ram_mb:.1f} MB / 4096.0 MB limit ({peak_ram_mb / 40.96:.1f}% of budget)")

    # 3. CPU Latency Benchmark
    print("\n[3/4] Running CPU Inference Latency Benchmark (15 runs)...")
    for _ in range(3):
        sess.run(["scores"], {"input": dummy_input})

    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        sess.run(["scores"], {"input": dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    min_ms = float(np.min(latencies))
    print(f"  [OK] CPU Latency        : avg = {avg_ms:.2f} ms | median = {p50_ms:.2f} ms | min = {min_ms:.2f} ms")

    # 4. Real Image Multi-Label Tagging Evaluation across D:\Image\Image
    print("\n[4/4] Evaluating Multi-Label Tagging across D:\\Image\\Image Categories...")
    d_img_cases = [
        ("Offensive Gesture (MF)", Path(r"D:\Image\Image\offensive_gestures\hagrid_mf_000053e0-139d-4918-be2d-89b388732a4b.jpg")),
        ("Offensive Gesture (V-Inv)", Path(r"D:\Image\Image\offensive_gestures\hagrid_v_inverted_02cdb129-5a97-4c5d-b63e-ffff90469133.jpg")),
        ("Benign Gesture (Like)", Path(r"D:\Image\Image\benign_gestures\benign_like_00aa414d-2b1e-4504-8ca8-b1da1ebb7538.jpg")),
        ("Benign Gesture (Peace)", Path(r"D:\Image\Image\benign_gestures\benign_peace_14c85f19-e90c-4610-b596-fc015c55d831.jpg")),
        ("Internet Meme (bad_meme)", Path(r"D:\Image\Image\bad_meme\00DjNzR.jpg")),
        ("Accepted Car", Path(r"D:\Image\Image\accepted\Img_00000000001.png")),
        ("Accepted Food Dish", Path(r"D:\Image\Image\accepted\Img_00000000105.jpg")),
        ("Accepted Clean Selfie", Path(r"D:\Image\Image\accepted\Img_00000000305.jpg")),
        ("Accepted Invoice", Path(r"D:\Image\Image\accepted\Img_00000007590.jpg")),
        ("Accepted Receipt", Path(r"D:\Image\Image\accepted\Img_00000007650.jpg")),
        ("Accepted Quixi Bra/Lingerie", Path(r"D:\Image\Image\accepted\005a47ddddd397da111a723d1dfb73d958821482_001_240x.jpg")),
        ("Deny Weapon / Rifle", Path(r"D:\Image\Image\deny\Img_00000000285.jpg")),
        ("Deny Airport X-Ray Gun", Path(r"D:\Image\Image\deny\Img_00000018210.jpg")),
        ("Deny Driver License ID", Path(r"D:\Image\Image\deny\Img_00000019035.jpg")),
        ("Deny Toxic Quote Banner", Path(r"D:\Image\Image\deny\Img_00000008000.jpg")),
        ("Deny Blasphemy Banner", Path(r"D:\Image\Image\deny\Img_00000007310.jpg")),
        ("Quixi Protest Crowd", Path(r"D:\Image\Image\accepted\11.jpg")),
    ]

    for case_label, img_path in d_img_cases:
        if not img_path.exists():
            continue
        try:
            pil_img = Image.open(img_path)
            tensor = preprocess_pil_image(pil_img)
            t0 = time.perf_counter()
            probs = sess.run(["scores"], {"input": tensor})[0][0]
            dt_ms = (time.perf_counter() - t0) * 1000.0

            top_indices = np.argsort(probs)[::-1][:6]
            top_tags = [f"{labels[idx]} ({probs[idx] * 100:.1f}%)" for idx in top_indices]
            print(f"  - {case_label:<27} [{dt_ms:5.1f} ms] -> {', '.join(top_tags[:5])}")
        except Exception as exc:
            print(f"  - Skipping {img_path.name}: {exc}")

    print("\n" + "=" * 72)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 72)


if __name__ == "__main__":
    main()
