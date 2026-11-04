# SAN-Web: Universal ONNX Multi-Label Image Tagger (Unity Sentis & ONNX Runtime Web)

Modello ONNX universale **All-in-One** per l'etichettatura multi-label di immagini (**4.660 tag in lingua inglese**), progettato per funzionare **sia in Unity con Sentis (`com.unity.sentis` / `com.unity.ai.inference`) sia nel browser con Microsoft ONNX Runtime (`onnxruntime-web`)** su CPU restando ampiamente sotto il limite di **4 GB di RAM** (~930 MB di picco).

---

## 📥 Clonare il Repository e Scaricare i Modelli con Git LFS

I file binari di grandi dimensioni (il modello `models/san_tagger.onnx` da ~398 MB e i pesi calibrati `models/d_image_trained_head.npz` da ~62 MB) sono gestiti tramite **Git LFS** (*Large File Storage*).

> [!IMPORTANT]
> Se si clona il repository con un semplice `git clone` senza avere Git LFS installato o attivo, al posto dei file binari effettivi verranno scaricati piccoli file puntatori testuali (pochi byte). Per avere i file completi al 100%, segui i passaggi seguenti:

### 1. Installazione di Git LFS (una tantum per macchina)
- **Windows**: Git LFS è già incluso nell'installer ufficiale di Git per Windows. Se necessario, esegui una volta:
  ```bash
  git lfs install
  ```
- **macOS** (Homebrew):
  ```bash
  brew install git-lfs && git lfs install
  ```
- **Linux** (Debian/Ubuntu):
  ```bash
  sudo apt-get install git-lfs && git lfs install
  ```

### 2. Clonare il repository scaricando subito i modelli
Eseguendo il clone con Git LFS attivo, tutti i pesi e il modello ONNX vengono scaricati automaticamente:
```bash
git clone https://github.com/SantoroNicola-dev/SAN-Web.git
cd SAN-Web
```

### 3. Se hai già clonato il repo (o per aggiornare / fare pull)
Se hai già fatto `git clone` o `git pull` ma i file in `models/` risultano vuoti o puntatori:
```bash
# Scarica/aggiorna i file tracciati da Git LFS per il branch corrente
git lfs pull
```

Per verificare che tutti i file LFS siano scaricati e integri:
```bash
git lfs ls-files
```

---

## 1. Caratteristiche Tecniche e Architettura del Grafo ONNX

Il file [`models/san_tagger.onnx`](models/san_tagger.onnx) racchiude l'intera pipeline in un unico grafo statico **ONNX Opset 15 FP32** senza dipendenze esterne:

1. **Input Standardizzato (`input`)**:
   - Tipo: `float32[1, 3, 224, 224]` (formato `NCHW`, canali `RGB` con valori in `[0.0, 1.0]`).
   - Compatibile direttamente con `TextureConverter.ToTensor` di Unity Sentis e con `ImageData` del Canvas HTML5.
2. **Normalizzazione Interna Integrata**:
   - Sottrazione `mean` (`[0.48145466, 0.4578275, 0.40821073]`) e divisione `std` (`[0.26862954, 0.26130258, 0.27577711]`) eseguite direttamente dentro i primi nodi `Sub` e `Div` del grafo ONNX.
3. **Backbone Visuale Static-Shape (ViT-B/32)**:
   - Costruito con LayerNorm e Self-Attention decomposti in sole **16 primitive matematiche elementari** (`Add`, `Concat`, `Constant`, `Conv`, `Div`, `Identity`, `MatMul`, `Mul`, `ReduceMean`, `ReduceSum`, `Reshape`, `Sigmoid`, `Softmax`, `Sqrt`, `Sub`, `Transpose`).
   - **Zero operatori INT8 dinamici** (`DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`) non supportati da Unity Sentis.
   - **Zero operatori di shape dinamico** (`Shape`, `Slice`, `ConstantOfShape`, `Gather`).
4. **Adattatore Visuale Residuale & Banca Prototipi Contrastiva (v3.0)**:
   - Residual Bottleneck Adapter (`512 -> 256 -> 512`) con attivazione QuickGELU per la separazione fine delle classi complesse.
   - Contrastive Prototype Memory Bank con 2.679 prototipi multi-dominio integrati nel grafo ONNX.
5. **Testa Multi-Label a 4.660 Tag + Calibrazione Sigmoid**:
   - Matrice di proiezione supervisionata `[512, 4660]` (vocabolario **Recognize Anything RAM 4.585 tag** + **categorie SAN-1**) con calibrazione empirica per-tag e attivazione `Sigmoid`.
6. **Output (`scores`)**:
   - Tipo: `float32[1, 4660]` — probabilità indipendente calibrata in `[0.0, 1.0]` per ciascuno dei 4.660 tag.

---

## 2. Specifiche e Risorse Misurate

| Metrica | Valore Misurato |
| :--- | :--- |
| **File Modello** | `models/san_tagger.onnx` (398.20 MB, file singolo autoconcluso) |
| **Dizionario Etichette** | `models/labels.json` & `models/labels.txt` (4.660 tag in inglese) |
| **ONNX Opset / Precisione** | Opset 15 / FP32 Puro |
| **RAM di Picco su CPU** | **929.6 MB** (22.7% del budget massimo di 4.096 MB) |
| **Latenza Inferenza CPU** | ~28–30 ms (Desktop ONNX Runtime CPU) |

---

## 3. Struttura del Progetto

```text
SAN-Web/
├── models/
│   ├── san_tagger.onnx          # Modello ONNX All-in-One v3.0 (Git LFS)
│   ├── d_image_trained_head.npz # Pesi supervisionati testa & prototipi v3.0 (Git LFS)
│   ├── calibration_image_vecs.npy # Embedding calibrazione multi-dominio (970 vettori)
│   ├── tag_embeddings_v2.npy    # Matrice ensemble 8 template neutral-centered [512, 4660]
│   ├── labels.json              # Metadati completi dei 4.660 tag inglesi
│   ├── labels.txt               # Lista testuale (1 tag per riga) per Unity TextAsset
│   └── model_manifest.json      # Checksum SHA-256, specifiche I/O e lista operatori
├── unity/
│   ├── SanImageTagger.cs        # Componente C# per Unity Inference Engine 2.x / Sentis
│   ├── SanImageTaggerDemo.cs    # Componente demo per inferenza immediata
│   ├── Editor/
│   │   └── SanTaggerSetup.cs    # Menu automatico "Tools -> SAN Tagger -> Create Tagger In Scene"
│   └── README_UNITY.md          # Guida passo-passo per Unity
├── web/
│   ├── index.html               # Web App interattiva con onnxruntime-web (WASM CPU)
│   ├── app.js                   # Pre-processing Canvas -> Tensor [1,3,224,224] -> Inferenza WASM
│   └── styles.css               # Stili dell'interfaccia web
├── scripts/
│   ├── build_tag_vocabulary.py  # Costruzione del dizionario a 4.660 tag (RAM + SAN-1)
│   ├── train_from_d_image.py    # Pipeline addestramento supervisionato adapter & prototipi
│   ├── export_onnx_model.py     # Esportazione del grafo ONNX All-in-One v3.0
│   └── verify_and_benchmark.py  # Test di compatibilità Sentis, memoria (<4GB) e benchmark CPU
├── .gitattributes               # Configurazione Git LFS per *.onnx e *.npz pesanti
├── .gitignore
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

## 5. Utilizzo in Unity (Inference Engine 2.x / Sentis)

1. Copia `models/san_tagger.onnx` e `models/labels.txt` in `Assets/Models/` nel tuo progetto Unity.
2. Copia i file da `unity/` nelle cartelle `Assets/Scripts/` e `Assets/Scripts/Editor/`.
3. Nella barra dei menu di Unity: seleziona **Tools -> SAN Tagger -> Create Tagger In Scene**.
4. Premi **Play**: il componente eseguirà l'inferenza e stamperà i tag nella Console.

Consulta [`unity/README_UNITY.md`](unity/README_UNITY.md) per tutti i dettagli su Unity Inference Engine (`com.unity.ai.inference`).

---

## 6. Rigenerazione e Verifica da Python

```bash
# 1. Rigenera il vocabolario di 4.660 tag (labels.json e labels.txt)
python scripts/build_tag_vocabulary.py

# 2. Esporta il modello ONNX All-in-One v3.0 (utilizza d_image_trained_head.npz)
python scripts/export_onnx_model.py

# 3. Esegui la suite di verifica (operatori Sentis, RAM < 4GB, latenza CPU, test immagini)
python scripts/verify_and_benchmark.py
```
