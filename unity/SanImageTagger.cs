// ============================================================================
// SanImageTagger.cs
// Universal Multi-Label Image Tagging Component for Unity Inference Engine
// (com.unity.ai.inference 2.x, formerly Sentis) - CPU / GPU.
//
// NOTE: the namespace is intentionally NOT "SanWeb.Unity": a namespace segment
// named "Unity" shadows the global "Unity.InferenceEngine" namespace and breaks
// compilation (error CS0234).
// ============================================================================

using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace SanWeb.Tagging
{
    [Serializable]
    public struct ImageTagPrediction
    {
        public int Index;
        public string Label;
        public float Confidence;      // 0.0 to 1.0
        public float ConfidencePct;   // 0.0% to 100.0%
    }

    public class SanImageTagger : MonoBehaviour
    {
        public const string DefaultModelPath = "Assets/Models/san_tagger.onnx";
        public const string DefaultLabelsPath = "Assets/Models/labels.txt";
        public const int InputResolution = 224;

        [Header("Model & Vocabulary Assets")]
        [Tooltip("Drag Assets/Models/san_tagger.onnx here")]
        public ModelAsset onnxModelAsset;

        [Tooltip("Drag Assets/Models/labels.txt here (1 English tag per line, 4660 lines)")]
        public TextAsset labelsTextAsset;

        [Header("Inference Settings")]
        [Tooltip("CPU is deterministic and stays well below 1 GB RAM. GPUCompute is also supported.")]
        public BackendType backend = BackendType.CPU;

        [Range(0.05f, 0.95f)]
        [Tooltip("Minimum confidence threshold (0.50 recommended for calibrated multi-label output)")]
        public float confidenceThreshold = 0.50f;

        [Range(1, 50)]
        [Tooltip("Maximum number of tags returned per image")]
        public int topK = 15;

        private Model _runtimeModel;
        private Worker _worker;
        private string[] _labels;
        private bool _isInitialized;

        public bool IsInitialized => _isInitialized;

        private void Awake()
        {
            Initialize();
        }

        /// <summary>
        /// Loads the ONNX model into the Inference Engine and parses the 4,660 English tags.
        /// </summary>
        public void Initialize()
        {
            if (_isInitialized) return;

            if (onnxModelAsset == null)
            {
                Debug.LogError("[SanImageTagger] Missing onnxModelAsset (san_tagger.onnx).", this);
                return;
            }

            if (labelsTextAsset == null)
            {
                Debug.LogError("[SanImageTagger] Missing labelsTextAsset (labels.txt).", this);
                return;
            }

            _labels = labelsTextAsset.text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            _runtimeModel = ModelLoader.Load(onnxModelAsset);
            _worker = new Worker(_runtimeModel, backend);

            _isInitialized = true;
            Debug.Log($"[SanImageTagger] Initialized on {backend} with {_labels.Length} tags.", this);
        }

        /// <summary>
        /// Classifies any Unity Texture (Texture2D, RenderTexture, WebCamTexture) and returns
        /// the top multi-label English tags sorted by confidence descending.
        /// Normalization ((x - mean) / std) and sigmoid calibration are baked into san_tagger.onnx,
        /// so the texture goes straight in as RGB [0..1], resized (stretched) to 224x224.
        /// sRGB textures are converted back to sRGB values automatically, so results match
        /// the browser / Python pipeline even in a Linear color-space project.
        /// </summary>
        public List<ImageTagPrediction> PredictTags(Texture sourceTexture, float? customThreshold = null, int? customTopK = null)
        {
            if (sourceTexture == null)
            {
                Debug.LogError("[SanImageTagger] PredictTags called with a null texture.", this);
                return new List<ImageTagPrediction>();
            }

            if (!_isInitialized)
            {
                Initialize();
                if (!_isInitialized) return new List<ImageTagPrediction>();
            }

            float minScore = customThreshold ?? confidenceThreshold;
            int maxResults = customTopK ?? topK;

            var transform = new TextureTransform()
                .SetDimensions(InputResolution, InputResolution, 3)
                .SetTensorLayout(TensorLayout.NCHW)
                .SetCoordOrigin(CoordOrigin.TopLeft);

            using Tensor<float> inputTensor = TextureConverter.ToTensor(sourceTexture, transform);
            _worker.Schedule(inputTensor);

            var outputTensor = _worker.PeekOutput("scores") as Tensor<float>;
            using Tensor<float> cpuScores = outputTensor.ReadbackAndClone();
            float[] scores = cpuScores.DownloadToArray();

            return ExtractTopPredictions(scores, minScore, maxResults);
        }

        private List<ImageTagPrediction> ExtractTopPredictions(float[] scores, float minScore, int maxResults)
        {
            int count = Mathf.Min(scores.Length, _labels.Length);
            var candidates = new List<ImageTagPrediction>(32);

            for (int i = 0; i < count; i++)
            {
                if (scores[i] >= minScore)
                {
                    candidates.Add(MakePrediction(i, scores[i]));
                }
            }

            // If no tag crossed minScore, still return the highest-scoring fallback tags
            if (candidates.Count == 0)
            {
                for (int i = 0; i < count; i++)
                {
                    if (scores[i] >= 0.25f)
                    {
                        candidates.Add(MakePrediction(i, scores[i]));
                    }
                }
            }

            candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
            if (candidates.Count > maxResults)
            {
                candidates.RemoveRange(maxResults, candidates.Count - maxResults);
            }

            return candidates;
        }

        private ImageTagPrediction MakePrediction(int index, float score)
        {
            return new ImageTagPrediction
            {
                Index = index,
                Label = _labels[index],
                Confidence = score,
                ConfidencePct = Mathf.Round(score * 1000f) / 10f
            };
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
            _worker = null;
            _isInitialized = false;
        }

#if UNITY_EDITOR
        // Auto-assign the default assets when the component is added in the Inspector.
        private void Reset()
        {
            AutoAssignDefaultAssets();
        }

        public void AutoAssignDefaultAssets()
        {
            if (onnxModelAsset == null)
                onnxModelAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(DefaultModelPath);
            if (labelsTextAsset == null)
                labelsTextAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>(DefaultLabelsPath);
        }
#endif
    }
}
