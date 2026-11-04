// ============================================================================
// SanTaggerSetup.cs (Editor only)
// Menu: Tools -> SAN Tagger -> Create Tagger In Scene
// Performs README step 3 automatically: creates a GameObject with
// SanImageTagger + SanImageTaggerDemo and wires the model, labels and a test image.
// ============================================================================

using SanWeb.Tagging;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SanWeb.Tagging.EditorTools
{
    public static class SanTaggerSetup
    {
        private const string TestImagePath = "Assets/TestImages/test_car.png";

        [MenuItem("Tools/SAN Tagger/Create Tagger In Scene")]
        public static void CreateTaggerInScene()
        {
            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(SanImageTagger.DefaultModelPath);
            var labels = AssetDatabase.LoadAssetAtPath<TextAsset>(SanImageTagger.DefaultLabelsPath);

            if (model == null || labels == null)
            {
                EditorUtility.DisplayDialog(
                    "SAN Tagger",
                    $"Asset mancanti.\nAttesi:\n- {SanImageTagger.DefaultModelPath}\n- {SanImageTagger.DefaultLabelsPath}",
                    "OK");
                return;
            }

            var existing = Object.FindFirstObjectByType<SanImageTagger>();
            GameObject go = existing != null ? existing.gameObject : new GameObject("SAN Image Tagger");
            if (existing == null) Undo.RegisterCreatedObjectUndo(go, "Create SAN Image Tagger");

            var tagger = go.GetComponent<SanImageTagger>();
            if (tagger == null) tagger = Undo.AddComponent<SanImageTagger>(go);
            Undo.RecordObject(tagger, "Configure SAN Image Tagger");
            tagger.onnxModelAsset = model;
            tagger.labelsTextAsset = labels;
            tagger.backend = BackendType.CPU;
            tagger.confidenceThreshold = 0.50f;
            tagger.topK = 15;

            var demo = go.GetComponent<SanImageTaggerDemo>();
            if (demo == null) demo = Undo.AddComponent<SanImageTaggerDemo>(go);
            Undo.RecordObject(demo, "Configure SAN Image Tagger Demo");
            if (demo.testImage == null)
                demo.testImage = AssetDatabase.LoadAssetAtPath<Texture2D>(TestImagePath);

            EditorUtility.SetDirty(tagger);
            EditorUtility.SetDirty(demo);
            EditorSceneManager.MarkSceneDirty(go.scene);
            Selection.activeGameObject = go;

            Debug.Log("[SanTaggerSetup] 'SAN Image Tagger' configurato. Salva la scena (Ctrl+S) e premi Play.");
        }
    }
}
