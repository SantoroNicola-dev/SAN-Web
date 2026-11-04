// ============================================================================
// SanImageTaggerDemo.cs
// Minimal test harness: tags `testImage` on Start (or via the component's
// context menu "Run Tagging") and prints the results to the Console.
// ============================================================================

using System.Text;
using UnityEngine;

namespace SanWeb.Tagging
{
    [RequireComponent(typeof(SanImageTagger))]
    public class SanImageTaggerDemo : MonoBehaviour
    {
        [Tooltip("Any Texture2D / RenderTexture to tag. Read/Write is NOT required.")]
        public Texture testImage;

        public bool runOnStart = true;

        [Range(0.05f, 0.95f)]
        public float threshold = 0.50f;

        [Range(1, 50)]
        public int topK = 15;

        private SanImageTagger _tagger;

        private void Start()
        {
            _tagger = GetComponent<SanImageTagger>();
            if (runOnStart) RunTagging();
        }

        [ContextMenu("Run Tagging")]
        public void RunTagging()
        {
            if (_tagger == null) _tagger = GetComponent<SanImageTagger>();
            if (testImage == null)
            {
                Debug.LogWarning("[SanImageTaggerDemo] Assign a Test Image in the Inspector.", this);
                return;
            }

            float t0 = Time.realtimeSinceStartup;
            var predictions = _tagger.PredictTags(testImage, threshold, topK);
            float ms = (Time.realtimeSinceStartup - t0) * 1000f;

            var sb = new StringBuilder();
            sb.AppendLine($"[SanImageTaggerDemo] '{testImage.name}' -> {predictions.Count} tags in {ms:F1} ms");
            foreach (var tag in predictions)
            {
                sb.AppendLine($"  {tag.Label}: {tag.ConfidencePct:F1}%");
            }
            Debug.Log(sb.ToString(), this);
        }
    }
}
