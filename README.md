# SAN-Web: Universal ONNX Multi-Label Image Tagger (Unity Sentis & ONNX Runtime Web)

Modello ONNX universale **All-in-One** per l'etichettatura multi-label di immagini (**4.660 tag in lingua inglese**), progettato per funzionare **sia in Unity con Sentis (`com.unity.sentis` / `com.unity.ai.inference`) sia nel browser con Microsoft ONNX Runtime (`onnxruntime-web`)** su CPU restando ampiamente sotto il limite di **4 GB di RAM** (~820 MB di picco).

---

## 1. Caratteristiche Tecniche e Architettura del Grafo ONNX

Il file [`models/san_tagger.onnx`](models/san_tagger.onnx) racchiude l'intera pipeline in un unico grafo statico **ONNX Opset 15 FP32** senza dipendenze esterne:

1. **Input Standardizzato (`input`)**:
   - Tipo: `float32[1, 3, 224, 224]` (formato `NCHW`, canali `RGB` con valori in `[0.0, 1.0]`).
   - Compatibile direttamente con `TextureConverter.ToTensor` di Unity Sentis e con `ImageData` del Canvas HTML5.
2. **Normalizzazione Interna Integrata**:
   - Sottrazione `mean` (`[0.48145466, 0.4578275, 0.40821073]`) e divisione `std` (`[0.26862954, 0.26130258, 0.27577711]`) eseguite direttamente dentro i primi nodi `Sub` e `Div` del grafo ONNX.
3. **Backbone Visuale Static-Shape (ViT-B/32)**:
   - Costruito con LayerNorm e Self-Attention decomposti in sole **15 primitive matematiche elementari** (`Add`, `Concat`, `Constant`, `Conv`, `Div`, `MatMul`, `Mul`, `ReduceMean`, `ReduceSum`, `Reshape`, `Sigmoid`, `Softmax`, `Sqrt`, `Sub`, `Transpose`).
   - **Zero operatori INT8 dinamici** (`DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`) non supportati da Unity Sentis.
   - **Zero operatori di shape dinamico** (`Shape`, `Slice`, `ConstantOfShape`, `Gather`).
4. **Testa Multi-Label a 4.660 Tag + Calibrazione Sigmoid**:
   - Matrice di proiezione pre-codificata `[512, 4660]` (combinazione del vocabolario **Recognize Anything RAM 4.585 tag** + **categorie SAN-1**) con calibrazione ibrida (similarità coseno assoluta + z-score statistico intra-immagine) e attivazione `Sigmoid`.
5. **Output (`scores`)**:
   - Tipo: `float32[1, 4660]` — probabilità indipendente calibrata in `[0.0, 1.0]` per ciascuno dei 4.660 tag.

---

## 2. Specifiche e Risorse Misurate

| Metrica | Valore Misurato |
| :--- | :--- |
| **File Modello** | `models/san_tagger.onnx` (344.30 MB, file singolo) |
| **Dizionario Etichette** | `models/labels.json` & `models/labels.txt` (4.660 tag in inglese) |
| **ONNX Opset / Precisione** | Opset 15 / FP32 Puro |
| **RAM di Picco su CPU** | **820.7 MB** (20.0% del budget massimo di 4.096 MB) |
| **Latenza Inferenza CPU** | ~25–95 ms (Desktop ONNX Runtime CPU) |

---

## 3. Struttura del Progetto

```text
SAN-Web/
├── models/
│   ├── san_tagger.onnx          # Modello ONNX All-in-One (FP32, Opset 15)
│   ├── labels.json              # Metadati completi dei 4.660 tag inglesi
│   ├── labels.txt               # Lista testuale (1 tag per riga) per Unity TextAsset
│   └── model_manifest.json      # Checksum SHA-256, specifiche I/O e lista operatori
├── unity/
│   ├── SanImageTagger.cs        # Componente C# per Unity Sentis 1.x e 2.x (Unity.InferenceEngine)
│   └── README_UNITY.md          # Guida passo-passo per Unity
├── web/
│   ├── index.html               # Web App interattiva con onnxruntime-web (WASM CPU)
│   ├── app.js                   # Pre-processing Canvas -> Tensor [1,3,224,224] -> Inferenza WASM
│   └── styles.css               # Stili dell'interfaccia web
├── scripts/
│   ├── build_tag_vocabulary.py  # Costruzione del dizionario a 4.660 tag (RAM + SAN-1)
│   ├── export_onnx_model.py     # Codifica multi-prompt ed esportazione del grafo ONNX
│   └── verify_and_benchmark.py  # Test di compatibilità Sentis, memoria (<4GB) e benchmark CPU
└── package.json
```

---

## 4. Utilizzo nel Browser (`onnxruntime-web` CPU)

Avvia un server HTTP locale dalla radice del progetto `SAN-Web`:

```bash
python -m http.server 8080
```

Apri nel browser:
👉 **`http://localhost:8080/web/`**

La Web App carica `models/san_tagger.onnx` tramite il provider `wasm` (CPU) di `onnxruntime-web` e consente di trascinare qualsiasi immagine per visualizzare in tempo reale i tag predetti e le percentuali di confidenza.

---

## 5. Utilizzo in Unity con Sentis

1. Copia `models/san_tagger.onnx` e `models/labels.txt` in `Assets/Models/` nel tuo progetto Unity.
2. Copia [`unity/SanImageTagger.cs`](unity/SanImageTagger.cs) in `Assets/Scripts/`.
3. Assegna `san_tagger.onnx` e `labels.txt` all'Inspector del componente `SanImageTagger` (con `BackendType.CPU`) e chiama:

```csharp
var tags = sanImageTagger.PredictTags(myTexture, customThreshold: 0.50f, customTopK: 15);
foreach (var t in tags)
{
    Debug.Log($"{t.Label}: {t.ConfidencePct:F1}%");
}
```

Consulta [`unity/README_UNITY.md`](unity/README_UNITY.md) per tutti i dettagli su Unity Sentis 1.x (`Unity.Sentis`) e Unity 6 AI (`Unity.InferenceEngine`).

---

## 6. Rigenerazione e Verifica da Python

```bash
# 1. Rigenera il vocabolario di 4.660 tag (labels.json e labels.txt)
python scripts/build_tag_vocabulary.py

# 2. Esporta il modello ONNX All-in-One (models/san_tagger.onnx)
python scripts/export_onnx_model.py

# 3. Esegui la suite di verifica (operatori Sentis, RAM < 4GB, latenza CPU, test immagini)
python scripts/verify_and_benchmark.py
```

