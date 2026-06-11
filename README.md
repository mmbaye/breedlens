<div align="center">

# 🌾 BreedLens

**Sorghum trait estimation from phone photos**

Automatic sorghum grain detection and counting with **YOLOv11**, mass and yield
estimation, and local agronomic analysis powered by a vision-language model
(**Ollama / Qwen3-VL**).

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933.svg)](https://nodejs.org/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com/)

</div>

---

## 🌍 Why BreedLens matters

Measuring grain number and estimating yield is one of the most time-consuming
steps in sorghum breeding programs. It is traditionally done by hand  counting
seeds, weighing samples, and recording results manually  a process that is slow,
tedious, and prone to human error. For breeders evaluating hundreds of genotypes
across multiple plots and seasons, this bottleneck directly limits how fast new,
better-adapted varieties can be developed.

BreedLens turns an ordinary smartphone into a field phenotyping tool. A single
photo is enough to count grains, estimate mass and yield, and produce a written
agronomic interpretation  in seconds, fully offline, with no specialized
equipment.

### What's new: pairing computer vision with a vision-language model

Most grain-counting tools stop at detection: they draw boxes and return a number.
BreedLens goes a step further by combining **two complementary AI layers**:

1. **Computer vision (YOLOv11)**  a specialized object-detection model trained
   to locate and count individual grains with high precision. This is the
   *measurement* layer: it produces objective, reproducible numbers (counts,
   bounding boxes, confidence scores).

2. **A vision-language model (Qwen3-VL via Ollama)**  a multimodal LLM that
   *looks at the same annotated image* together with the detection figures and
   produces a natural-language agronomic interpretation: grain distribution and
   uniformity, visible anomalies, yield context, and recommendations. This is the
   *reasoning* layer.

This pairing is what makes BreedLens novel. The detection model answers
**"how many?"** with numerical rigor, while the vision-language model answers
**"what does this mean agronomically?"** in plain language a breeder or
technician can act on. Instead of a raw count, the user gets a measurement **and**
an expert-style reading of the sample — and because both models run locally
through Ollama and YOLO, it all works in the field, without internet, and without
sending a single image to the cloud.

---

## 📋 Overview

**BreedLens** is a local web application designed for breeders, laboratory
technicians, and researchers working on sorghum. From a single phone photo
(a panicle or spread grains), it:

- automatically detects and counts grains using a **YOLOv11** model;
- estimates **mass** and **yield** from adjustable parameters;
- generates a natural-language **agronomic analysis** (English or French);
- processes images in **batch** and exports a ready-to-use **CSV**.

> 🔒 **100% local** — detection and analysis run on your own machine. No image
> is ever sent to a third-party server.

---

## ✨ Features

| Feature                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 📷 **Single detection**   | Photo (rear camera) or gallery import → annotated image + statistics        |
| 📂 **Batch processing**   | Multiple images at once → summary table + **CSV** export                     |
| ⚖️ **Mass estimation**    | Mass and yield computation from calibratable parameters                      |
| 🤖 **Agronomic analysis** | Automatic interpretation of the annotated image (Ollama / Qwen3-VL)          |
| 🌍 **Bilingual**          | Interface and analysis in **English** (default) or **French**               |
| 📱 **Mobile / PWA**       | Usable from a phone or iPad, installable on the home screen                  |

---

## 🏗️ Architecture

The application relies on three local services that communicate with each other:

```
              Browser  (index.html)
                    │
                    │  HTTP  ·  port 3001
                    ▼
            server.js  (Node / Express)
            relay + Ollama analysis
              │                    │
   port 5001  │                    │  port 11434
              ▼                    ▼
      yolo_server.py          Ollama  (Qwen3-VL)
        (Flask)
              │
              ▼
   predict.py  +  YOLOv11 model (.pt)
```

| Component         | Role                                                            | Port    |
|-------------------|-----------------------------------------------------------------|---------|
| `index.html`      | Web interface (capture, batch, result display)                  | —       |
| `server.js`       | Node server: serves the UI, relays to YOLO, calls Ollama        | `3001`  |
| `yolo_server.py`  | Flask server: YOLO detection (single and batch)                 | `5001`  |
| `predict.py`      | YOLOv11 inference script (single detection)                     | —       |
| Ollama            | Vision-language model for the agronomic analysis                | `11434` |

> ℹ️ Batch detection loads the YOLO model **once** in memory for the entire
> batch (faster than relaunching `predict.py` per image), while keeping the same
> counting logic as `predict.py`.

---

## 🔧 Requirements

- **Node.js** 18 or higher
- **Python** 3.10+ (Conda recommended) with `ultralytics`, `opencv`, `flask`, `flask-cors`
- **Ollama** installed, with the vision model downloaded
- The **trained YOLOv11 model** (`.pt` file)

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/mmbaye/breedlens.git
cd breedlens
```

### 2. Node dependencies

```bash
npm install express cors dotenv
```

### 3. Python environment (Conda)

```bash
conda create -n yolo python=3.10 -y
conda activate yolo
pip install ultralytics opencv-python flask flask-cors
```

> ⚠️ **Always** run the Python server from the environment where these packages
> are installed. If you get `ModuleNotFoundError: No module named 'flask_cors'`,
> the Conda `yolo` environment is not activated.

### 4. Ollama and vision model

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3-vl:8b
```

### 5. Place the YOLO model

Copy your weights file (`.pt`) to a known location, then set its path in
`yolo_server.py` (see [Configuration](#️-configuration)).

---

## ⚙️ Configuration

### `yolo_server.py`

At the top of the file, adapt these variables to your machine:

```python
PYTHON_BIN  = "/opt/anaconda3/envs/yolo/bin/python"      # python from the yolo env
PREDICT_PY  = str(Path(__file__).parent / "predict.py")
MODEL_PATH  = "/path/to/grain_detection_yolom.pt"        # your .pt model
```

Flask server port (bottom of the file):

```python
app.run(host='127.0.0.1', port=5001, debug=False)
```

### `server.js`

Node server listening port (last line):

```javascript
app.listen(3001, '0.0.0.0', () => { ... });
```

The `/detect` and `/detect-batch` routes must point to the **same port** as the
Flask server:

```javascript
fetch('http://127.0.0.1:5001/detect', ...)
fetch('http://127.0.0.1:5001/detect-batch', ...)
```

> 🔑 **Golden rule**: the port in `app.run(...)` of `yolo_server.py` and the
> `fetch('http://127.0.0.1:PORT/...')` calls in `server.js` must be **identical**.
> A mismatch produces the `Unexpected token '<'` error (the browser receives an
> HTML error page instead of the expected JSON).

### Estimation parameters (adjustable in the interface)

| Parameter        | Default  | Meaning                                              |
|------------------|----------|------------------------------------------------------|
| Slope (g/grain)  | `0.026`  | Average weight per grain                             |
| Face factor      | `2.0`    | "Visible face" correction (to be calibrated manually) |
| Density (pl/ha)  | `150000` | Actual planting density of the trial                 |

> ⚠️ The model detects only **one face** of the panicle. Face factor = 2.0 is a
> default assumption, to be adjusted after calibration with manual counting.

---

## ▶️ Running the application

Open three terminals (or use the launch script below):

```bash
# 1. Ollama (if not already running as a service)
ollama serve

# 2. YOLO server (conda environment activated)
conda activate yolo
python yolo_server.py        # → http://127.0.0.1:5001

# 3. Node server
node server.js               # → http://localhost:3001
```

Then open **http://localhost:3001** in your browser.

### Optional launch script (`start_yolo.sh`)

```bash
#!/bin/bash
conda activate yolo
python yolo_server.py
```

```bash
chmod +x start_yolo.sh
./start_yolo.sh
```

---

## 📖 Usage

### Single detection

1. Choose the language (🇬🇧 EN by default, 🇫🇷 FR available).
2. **Take a photo** (rear camera) or **Import** from the gallery.
3. Adjust the estimation parameters if needed.
4. Click **Detect & estimate**.
5. Review the annotated image, grain count, mass, confidence, estimated yield,
   then the agronomic analysis.

### Batch processing

1. In **Batch processing**, click **Choose multiple images**.
2. Select several images.
3. Each image is analyzed → a summary table is displayed.
4. Click **Download CSV** to retrieve the file.

> Batch processing uses the **Slope** and **Face factor** parameters set in the
> interface: adjust them **before** running the batch.

---

## 🤖 Agronomic analysis prompt

When you click **Detect & estimate**, BreedLens sends the **annotated image**
plus the detection figures to the Qwen3-VL vision-language model through Ollama.
The prompt is generated automatically in `server.js` (route `/analyze-sorghum`)
and adapts to the selected language. Below is the English version of the prompt
the model receives (placeholders such as `${count}` are filled with the actual
detection values):

```text
You are analyzing a sorghum panicle image with automatic grain detection
by YOLOv11.

Detection data:
- Grains detected (visible face only): ${count}
- Face correction factor: ${face_factor} (default value — not yet manually
  calibrated)
- Estimated total grains: ${grains_total} (= ${count} × ${face_factor})
- Estimated mass: ${mass_g} g (slope: ${slope_used} g/grain)
- Mean confidence: ${conf_mean}%
- Plant density: ${plant_density} plants/ha
- Estimated yield: ${yield_t_ha} t/ha
- Detected classes: ${class_counts}

Important note: The YOLOv11 model detects only one face of the panicle.
The factor ${face_factor} is a default assumption. These estimates should be
validated by manual counting.

Agronomic analysis requested:
1. Evaluate the density and distribution of grains on the visible face
2. Comment on apparent grain quality (size, uniformity, shape)
3. Identify any visible anomalies (empty zones, missing grains, disease or
   stress signs)
4. Provide an agronomic interpretation of the estimated yield
5. Formulate recommendations to improve estimation accuracy
```

> The French version follows the same structure and is selected automatically
> when the interface language is set to **FR**. The prompt is centralized in
> `server.js`, so any wording change is made in a single place.

---

## 📊 CSV format (batch)

One row per image:

| Column         | Description                                          |
|----------------|------------------------------------------------------|
| `image_id`     | Image file name (identifier)                         |
| `count`        | Number of detected grains (visible face)             |
| `grains_total` | `count` × face factor                                |
| `mass_g`       | Estimated mass in grams                              |
| `conf_mean`    | Mean detection confidence                            |
| `conf_min`     | Minimum confidence                                   |
| `conf_max`     | Maximum confidence                                   |
| `class_counts` | Per-class count, format `class:n`                    |
| `status`       | `ok`, or an error message if the image failed        |

---

## 📱 Using from an iPad / phone

The interface is accessible from a tablet or phone connected to the **same
Wi-Fi network** as the machine running the servers (all computation stays on the
machine).

1. Confirm the server listens on all interfaces in `server.js`:
   `app.listen(3001, '0.0.0.0', ...)`.
2. Find the machine's local IP address:
   ```bash
   ipconfig getifaddr en0      # macOS
   ```
3. On the iPad, open Safari: `http://<machine-IP>:3001`.
4. For an app-like experience: **Share → Add to Home Screen** (the app opens
   full screen thanks to the PWA manifest).

> If the camera won't open over HTTP on the local network, gallery import still
> works. To enable the camera over HTTPS, a tunnel such as `ngrok http 3001`
> provides a temporary https URL.

---

## 🛠️ Troubleshooting

| Symptom                                             | Likely cause / fix                                                               |
|-----------------------------------------------------|----------------------------------------------------------------------------------|
| `Unexpected token '<'`                              | Mismatched ports between `server.js` and `yolo_server.py`, or server not restarted. Align the ports and relaunch. |
| `ModuleNotFoundError: No module named 'flask_cors'` | Conda `yolo` environment not activated. Run `conda activate yolo` then `pip install flask-cors`. |
| `YOLO server unavailable`                           | `yolo_server.py` is not running, or not on the expected port.                  |
| `Ollama unavailable`                                | Ollama is not running (`ollama serve`) or the model is not downloaded.          |
| Interface not updated after editing                 | Browser cache: force-reload (`Cmd+Shift+R`).                                    |
| No detection on a sharp image                       | Check `MODEL_PATH` and the confidence threshold.                               |

---

## 📁 Project structure

```
breedlens/
├── index.html        # Web interface (capture, batch, results)
├── server.js         # Node server: UI + YOLO relay + Ollama
├── yolo_server.py    # Flask server: YOLO detection (single + batch)
├── predict.py        # YOLOv11 inference script (single detection)
├── manifest.json     # PWA manifest (home screen installation)
├── README.md         # This file
└── LICENSE           # MIT license
```

---

## 📝 Notes

- No image is transmitted to an external server: detection (YOLO) and analysis
  (Ollama) run locally.
- Mass and yield estimates depend on parameters that must be calibrated (slope,
  face factor, density). The default values are starting points and should be
  validated with manual counting in the field.

---

## 📄 License

Distributed under the **MIT** License. See the [`LICENSE`](LICENSE) file for
details.

---

<div align="center">

Built for sorghum research and variety selection 🌾

</div>
