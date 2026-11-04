# Integrazione in Unity con Inference Engine (`com.unity.ai.inference`, ex Sentis)

Il modello `models/san_tagger.onnx` è progettato per essere importato nativamente in **Unity Inference Engine 2.x** senza nodi incompatibili:
- **Opset ONNX 15** con soli 16 operatori elementari (`Add`, `Concat`, `Constant`, `Conv`, `Div`, `Identity`, `MatMul`, `Mul`, `ReduceMean`, `ReduceSum`, `Reshape`, `Sigmoid`, `Softmax`, `Sqrt`, `Sub`, `Transpose`).
- **Zero operatori INT8 dinamici** (`DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`).
- **Zero operatori di shape dinamico** (`Shape`, `Slice`, `ConstantOfShape`).
- **Normalizzazione e calibrazione integrate nel grafo**: `TextureConverter.ToTensor` passa direttamente un tensore RGB `[0..1]` `[1, 3, 224, 224]` e riceve le probabilità multi-label `[0..1]` `[1, 4660]`.

Testato con **Unity 6000.4** + **`com.unity.ai.inference` 2.6.1** (URP, Linear color space).

## Passaggi di configurazione in Unity

1. **Installa il pacchetto**:
   - `Window -> Package Manager -> + -> Add package by name...` -> `com.unity.ai.inference`.
   - Non serve alcun Scripting Define Symbol.
2. **Importa gli asset nel progetto Unity** (i percorsi sono importanti, lo script li usa per l'auto-assegnazione):
   - `models/san_tagger.onnx` -> `Assets/Models/san_tagger.onnx`
   - `models/labels.txt` -> `Assets/Models/labels.txt`
   - `unity/SanImageTagger.cs` e `unity/SanImageTaggerDemo.cs` -> `Assets/Scripts/`
   - `unity/Editor/SanTaggerSetup.cs` -> `Assets/Scripts/Editor/` (deve stare in una cartella chiamata `Editor`)
   - Attendi che Unity finisca l'import (l'ONNX è ~400 MB) e verifica che la **Console non mostri errori `CS`**: finché lo script non compila, il componente non può essere aggiunto.
3. **Collega il componente a un GameObject**

   **Opzione A – automatica (consigliata)**: menu `Tools -> SAN Tagger -> Create Tagger In Scene`, poi `Ctrl+S` per salvare la scena. Crea il GameObject `SAN Image Tagger` già configurato come nell'opzione B.

   **Opzione B – manuale**:
   1. Nella finestra **Hierarchy**: tasto destro -> `Create Empty`, rinominalo `SAN Image Tagger`.
   2. Con il GameObject selezionato, nell'**Inspector**: `Add Component` -> cerca `San Image Tagger` -> aggiungilo.
      Grazie a `Reset()`, i campi **Onnx Model Asset** e **Labels Text Asset** vengono compilati da soli se gli asset sono nei percorsi del punto 2. Altrimenti trascinali dalla finestra **Project**:
      - `Assets/Models/san_tagger.onnx` -> campo **Onnx Model Asset** (l'icona dell'asset è quella del modello, non un file generico)
      - `Assets/Models/labels.txt` -> campo **Labels Text Asset**
   3. Imposta i campi:
      | Campo | Valore | Note |
      | :--- | :--- | :--- |
      | **Backend** | `CPU` | deterministico, ~1 GB di RAM; `GPUCompute` è supportato |
      | **Confidence Threshold** | `0.50` | soglia calibrata per l'uscita multi-label |
      | **Top K** | `15` | numero massimo di tag restituiti |
   4. (Facoltativo, per un test) `Add Component` -> `San Image Tagger Demo`, trascina un'immagine nel campo **Test Image** (es. `Assets/TestImages/test_car.png`). Non serve abilitare *Read/Write* sulla texture.
   5. Premi **Play**: nella Console compaiono `[SanImageTagger] Initialized on CPU with 4660 tags.` e la lista dei tag. In Play Mode puoi anche rilanciare il test dal menu ⋮ del componente demo -> **Run Tagging**.
4. **Chiama il modello da C#**:
   ```csharp
   using SanWeb.Tagging;

   public class MyScript : MonoBehaviour
   {
       public SanImageTagger tagger;   // trascina qui il GameObject "SAN Image Tagger"
       public Texture2D myTexture2D;

       void Start()
       {
           var predictions = tagger.PredictTags(myTexture2D, customThreshold: 0.50f, customTopK: 15);
           foreach (var tag in predictions)
           {
               Debug.Log($"{tag.Label}: {tag.ConfidencePct:F1}%");
           }
       }
   }
   ```

## Note tecniche
- Il namespace è `SanWeb.Tagging` (e non `SanWeb.Unity`): un segmento di namespace chiamato `Unity` nasconde il namespace globale `Unity.InferenceEngine` e causa l'errore `CS0234`.
- L'immagine viene ridimensionata (stirata, senza crop) a 224×224, come nella demo browser e negli script Python.
- Le texture sRGB vengono riconvertite automaticamente in valori sRGB da `TextureConverter`, quindi i risultati coincidono con browser/Python anche in un progetto in Linear color space.
