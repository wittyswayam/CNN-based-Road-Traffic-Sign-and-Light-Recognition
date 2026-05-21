<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=F7A800&center=true&vCenter=true&width=700&lines=CNN+Road+Traffic+Recognition;Transfer+Learning+%7C+ResNet50;Bounding+Box+Regression+%7C+TensorFlow" alt="Typing SVG" />

# 🚦 CNN-based Road Traffic Sign & Light Recognition

**Real-time localization of road traffic signs and lights using Transfer Learning and Bounding Box Regression**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-ResNet50-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition?style=for-the-badge&logo=github)](https://github.com/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition/stargazers)
[![Forks](https://img.shields.io/github/forks/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition?style=for-the-badge&logo=github)](https://github.com/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition/network/members)

<br/>

[🔍 Overview](#-overview) •
[✨ Features](#-features) •
[🧠 Architecture](#-model-architecture) •
[📊 Dataset](#-dataset) •
[⚙️ Setup](#️-setup) •
[🚀 Usage](#-usage) •
[📈 Results](#-results) •
[🗺️ Roadmap](#️-roadmap) •
[🤝 Contributing](#-contributing) •
[📚 Citation](#-citation)

<br/>

> *A computer vision pipeline for detecting and localizing road traffic objects — built as a foundational perception module for autonomous driving and intelligent transport systems.*

</div>

---

## 📌 Overview

This project implements an end-to-end **deep learning pipeline** for the **detection and spatial localization** of road traffic signs and lights in images. Using **Transfer Learning** with a **ResNet50** backbone pretrained on ImageNet, the system is fine-tuned with a custom regression head to predict precise **bounding box coordinates** directly from raw image input.

The task is formulated as **Bounding Box Regression**: given an input image, the model outputs four normalized coordinates — `[x_min, y_min, x_max, y_max]` — that describe the location of the traffic object within the image. This localization capability is a foundational building block for higher-order autonomous systems that require reliable scene understanding.

### 🎯 Motivation

Safe autonomous driving depends on a vehicle's ability to accurately perceive and interpret its environment in real time. Traffic signs and lights are among the most critical cues in road environments — they encode speed limits, hazard warnings, intersection rules, and right-of-way decisions. A failure to detect them reliably can have life-threatening consequences.

This project demonstrates:

- How **pretrained convolutional feature extractors** can be repurposed for object localization with minimal labeled data
- How **YOLO-format annotations** can be parsed and used to supervise a regression task
- How **coordinate prediction** from a neural network can be rescaled and rendered back onto original-resolution images for interpretable visual output

### 🌐 Applications

| Domain | Use Case |
|---|---|
| Autonomous Vehicles | Real-time traffic sign perception for self-driving cars |
| Driver Assistance | ADAS systems alerting drivers to missed or misread signs |
| Smart City Infrastructure | Automated traffic monitoring and violation detection |
| Fleet Management | Dashcam-based post-hoc compliance auditing |
| Research & Benchmarking | Baseline localization model for comparative CV studies |

---

## ✨ Features

### Core Capabilities

- **Transfer Learning with ResNet50** — Leverages deep representations from a network trained on 1.2M ImageNet images, drastically reducing data requirements for convergence
- **Bounding Box Regression** — Predicts four continuous, normalized coordinates `[x_min, y_min, x_max, y_max]` using a sigmoid-activated output head constrained to `[0, 1]`
- **YOLO-Format Label Parsing** — Robust preprocessing function that reads YOLO-style `.txt` annotation files, extracts bounding box coordinates, and handles malformed or missing labels gracefully with `[0, 0, 0, 0]` placeholders
- **Image Preprocessing Pipeline** — Consistent resizing to `224×224`, BGR-to-float32 casting, and per-pixel normalization to `[0, 1]` for stable gradient flow
- **Coordinate Rescaling at Inference** — `predict_and_visualize()` maps normalized predictions back to original image pixel space with boundary clamping (`max(0, ...)`, `min(width, ...)`) to prevent out-of-bounds boxes
- **Visual Ground Truth Verification** — `display_random_image_with_bbox()` renders green bounding boxes on randomly sampled training images for quick data sanity checks
- **Structured Train/Test Split** — 80/20 split via `sklearn.model_selection.train_test_split` with fixed `random_state=42` for reproducibility
- **Kaggle-Ready** — Dataset paths pre-configured for Kaggle Notebooks; documented for local adaptation

### Technical Highlights

| Aspect | Detail |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained, fully frozen) |
| Custom Head | GlobalAveragePooling2D → Dense(1024, ReLU) → Dense(4, Sigmoid) |
| Optimizer | Adam, learning rate = 1e-4 |
| Loss | Mean Squared Error (MSE) over 4 coordinate outputs |
| Input Resolution | 224 × 224 × 3 (RGB) |
| Output | 4-dimensional normalized bounding box vector |
| Training Split | 80% train / 20% test |
| Batch Size | 32 |
| Epochs | 10 |
| Framework | TensorFlow 2.x / Keras |

---

## 🧠 Model Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT PIPELINE                        │
│                                                          │
│  Raw Image (any resolution)                              │
│       │                                                  │
│       ▼                                                  │
│  cv2.resize → (224, 224)                                 │
│       │                                                  │
│       ▼                                                  │
│  Normalize: pixel / 255.0   →   float32 ∈ [0.0, 1.0]   │
│       │                                                  │
│       ▼                                                  │
│  np.expand_dims → (1, 224, 224, 3)   [batch dim]        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              ResNet50 BACKBONE (Frozen)                  │
│                                                          │
│  Input: (None, 224, 224, 3)                              │
│                                                          │
│  Conv1 → BN → ReLU → MaxPool                             │
│       │                                                  │
│  Residual Block Group 1  (64 filters)                    │
│       │                                                  │
│  Residual Block Group 2  (128 filters)                   │
│       │                                                  │
│  Residual Block Group 3  (256 filters)                   │
│       │                                                  │
│  Residual Block Group 4  (512 filters)                   │
│       │                                                  │
│  Output Feature Map: (None, 7, 7, 2048)                  │
│                                                          │
│  ⚠ base_model.trainable = False                          │
│    All 25M+ ResNet50 parameters are frozen.              │
│    Gradients flow only through the custom head.          │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              CUSTOM REGRESSION HEAD                      │
│                                                          │
│  GlobalAveragePooling2D                                  │
│    (None, 7, 7, 2048) → (None, 2048)                    │
│       │                                                  │
│  Dense(1024, activation='relu')                          │
│    (None, 2048) → (None, 1024)                           │
│       │                                                  │
│  Dense(4, activation='sigmoid')                          │
│    (None, 1024) → (None, 4)                              │
│                                                          │
│  Output: [x_min, y_min, x_max, y_max] ∈ [0.0, 1.0]     │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              INFERENCE & RENDERING                       │
│                                                          │
│  Predicted normalized coords                             │
│       │                                                  │
│       ▼                                                  │
│  Scale to original image dimensions:                     │
│    x_min_px = int(x_min * original_width)               │
│    y_min_px = int(y_min * original_height)               │
│    x_max_px = int(x_max * original_width)               │
│    y_max_px = int(y_max * original_height)               │
│       │                                                  │
│       ▼                                                  │
│  Clamp to image bounds:                                  │
│    x_min = max(0, x_min_px)                              │
│    x_max = min(original_width, x_max_px)                 │
│       │                                                  │
│       ▼                                                  │
│  cv2.rectangle → Green bounding box overlay              │
│  plt.imshow → Rendered visualization                     │
└──────────────────────────────────────────────────────────┘
```

### Why ResNet50?

ResNet50's **skip connections** (residual shortcuts) solve the vanishing gradient problem that plagued deep networks before 2015. With 50 layers and 25M+ parameters trained on ImageNet's 1,000-class dataset, it has learned a rich hierarchy of visual features — edges, textures, shapes, object parts — that transfer well to traffic object localization without requiring large domain-specific labeled datasets.

| Property | Value |
|---|---|
| Architecture | Residual Network (He et al., 2015) |
| Depth | 50 layers |
| Parameters | ~25.6M total (frozen in this project) |
| Pretrain Dataset | ImageNet (1.28M images, 1,000 classes) |
| Output Feature Map | 7 × 7 × 2048 (before pooling) |
| Keras Import | `tensorflow.keras.applications.ResNet50` |

### Why Sigmoid Output Activation?

The ground truth bounding box labels are in YOLO-normalized format: all four coordinates are floats in `[0, 1]`. Using `sigmoid` in the output layer directly constrains predictions to this same range, eliminating the need for post-processing clipping and ensuring coordinate validity at inference time.

### Why MSE Loss?

Bounding box regression is a **continuous prediction** problem — the model outputs four real-valued coordinates, not class probabilities. Mean Squared Error directly measures the average squared deviation between predicted and ground-truth coordinates:

```
MSE = (1/4) * Σ (predicted_coord_i - true_coord_i)²
```

This penalizes larger positional errors more aggressively (squared), encouraging the model to avoid gross mislocalization.

---

## 📊 Dataset

### Dataset Overview

| Property | Details |
|---|---|
| **Name** | Car Detection Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/your_kaggle_username/cardetection) |
| **Format** | YOLO — paired `images/` (JPEG/PNG) and `labels/` (TXT) directories |
| **Image Type** | JPEG / PNG |
| **Label Format** | YOLO: `class_id x_min y_min x_max y_max` (normalized, space-separated) |
| **Input Size** | Resized to 224 × 224 before training |
| **Normalization** | Pixel values ÷ 255.0 → `[0.0, 1.0]` |
| **Train/Test Split** | 80% / 20% (sklearn, `random_state=42`) |

### Directory Structure (Kaggle)

```
/kaggle/input/cardetection/
└── car/
    ├── train/
    │   ├── images/
    │   │   ├── 00000_00000_00016_png.rf.b8f0678f2b179f3b8d50d47b1549b069.jpg
    │   │   ├── 00001_00001_00032_png.rf.*.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── 00000_00000_00016_png.rf.b8f0678f2b179f3b8d50d47b1549b069.txt
    │       └── ...
    └── test/
        ├── images/
        │   ├── 000006_jpg.rf.89610ec419ccfab22f8314026b90ee26.jpg
        │   └── ...
        └── labels/
            └── ...
```

### Label File Format

Each `.txt` file contains one annotation per line. The pipeline parses the first five tokens:

```
0 0.3124 0.2045 0.7891 0.8102
│    │      │      │      │
│    │      │      │      └── y_max  (normalized, 0–1)
│    │      │      └───────── x_max  (normalized, 0–1)
│    │      └──────────────── y_min  (normalized, 0–1)
│    └─────────────────────── x_min  (normalized, 0–1)
└──────────────────────────── class_id (skipped by loader)
```

> **Note:** The class ID is intentionally skipped. This version of the model performs localization only — it predicts *where* the object is, not *what* the object is.

### Preprocessing Pipeline — Step by Step

```
For each image file in images/:
  1. Load with cv2.imread()                    → BGR uint8 array
  2. Resize: cv2.resize(img, (224, 224))       → (224, 224, 3) uint8
  3. Cast: img.astype('float32')               → (224, 224, 3) float32
  4. Normalize: img / 255.0                    → values ∈ [0.0, 1.0]
  5. Append to images list

For each matching label file in labels/:
  6. Read and strip whitespace
  7. Split on whitespace → list of strings
  8. Skip token[0] (class_id)
  9. Parse tokens[1:5] → [x_min, y_min, x_max, y_max] as float
 10. If len(tokens) < 5: use placeholder [0, 0, 0, 0]
 11. If label file missing: use placeholder [0, 0, 0, 0]

Final arrays:
  X: np.array of shape (N, 224, 224, 3)   dtype=float32
  y: np.array of shape (N, 4)              dtype=float64
```

### Data Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42      # Fixed seed for reproducibility
)
```

---

## 🗂️ Project Structure

```
CNN-based-Road-Traffic-Sign-and-Light-Recognition/
│
├── 📓 CNN-based-Road-Traffic-Sign-and-Light-Recognition.ipynb
│       The main Jupyter Notebook. Contains the complete pipeline:
│       data loading, preprocessing, visualization, model definition,
│       training, evaluation, and inference.
│
├── 📄 README.md
│       This file.
│
├── 📄 LICENSE
│       MIT License.
│
├── 📁 assets/                         (create this manually)
│   ├── sample_gt_bbox.png             Ground truth bounding box sample
│   ├── sample_prediction.png          Predicted bounding box sample
│   └── architecture_diagram.png       Model architecture visual
│
├── 📁 data/                           (not tracked — see .gitignore)
│   └── car/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
│
└── 📄 .gitignore
```

> **Note:** The `data/` directory and any trained model weights (`.h5`, `SavedModel`) should be added to `.gitignore` and not committed to the repository.

### Recommended `.gitignore`

```gitignore
# Dataset
data/

# Model weights
*.h5
saved_model/
*.keras

# Python cache
__pycache__/
*.pyc
*.pyo

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment
.venv/
venv/
env/

# System
.DS_Store
Thumbs.db
```

---

## ⚙️ Setup

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB | 10 GB |
| GPU | Optional (CPU works) | NVIDIA CUDA-capable |
| OS | Linux / macOS / Windows | Ubuntu 22.04 |

---

### Option A — `uv` (Recommended — Fastest)

[`uv`](https://github.com/astral-sh/uv) is a Rust-based Python package manager that is 10–100× faster than pip.

**Step 1: Install `uv`**

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 2: Clone the repository**

```bash
git clone https://github.com/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition.git
cd CNN-based-Road-Traffic-Sign-and-Light-Recognition
```

**Step 3: Create a virtual environment**

```bash
uv venv
```

**Step 4: Activate the environment**

```bash
# Linux / macOS
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Step 5: Install dependencies**

```bash
uv pip install tensorflow numpy opencv-python scikit-learn matplotlib jupyter
```

**Step 6: (Optional) GPU support**

```bash
# Install CUDA-enabled TensorFlow
uv pip install tensorflow[and-cuda]
```

---

### Option B — `pip` (Standard)

```bash
# Clone
git clone https://github.com/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition.git
cd CNN-based-Road-Traffic-Sign-and-Light-Recognition

# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install --upgrade pip
pip install tensorflow numpy opencv-python scikit-learn matplotlib jupyter
```

---

### Option C — `requirements.txt`

Create a `requirements.txt` for locked, reproducible installs:

```txt
tensorflow>=2.13.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

Then install:

```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

---

### Kaggle API Setup (Dataset Download)

**Step 1: Install the Kaggle CLI**

```bash
pip install kaggle
# or
uv pip install kaggle
```

**Step 2: Get your API token**

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Click **"Create New Token"** — this downloads `kaggle.json`

**Step 3: Place the credentials file**

```bash
# Linux / macOS
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json       # Restrict file permissions

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\kaggle.json
```

> ⚠️ **Security:** Never commit `kaggle.json` to version control. Add `kaggle.json` to your `.gitignore`.

**Step 4: Download and extract the dataset**

```bash
mkdir -p data
cd data

# Download the dataset (replace with the actual Kaggle dataset slug)
kaggle datasets download -d your_kaggle_username/cardetection

# Extract
unzip cardetection.zip
```

**Step 5: Verify the directory structure**

```bash
ls data/car/train/
# Expected: images/  labels/

ls data/car/train/images/ | head -5
# Expected: *.jpg or *.png files
```

---

### Jupyter Notebook Setup

```bash
# Ensure your virtual environment is activated, then:
jupyter notebook

# Or JupyterLab (modern interface)
pip install jupyterlab
jupyter lab
```

Navigate to `CNN-based-Road-Traffic-Sign-and-Light-Recognition.ipynb` in the browser and open it.

**Using Kaggle's free cloud environment (no local setup needed):**

1. Upload the notebook at [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. Add the dataset via **"Add Data"** → search for `cardetection`
3. Enable GPU under **Session Options → Accelerator → GPU T4 × 2**
4. Click **"Run All"**

---

## 🚀 Usage

### Step 1 — Configure Dataset Path

Open the notebook and locate the dataset path cell. Update it to match your environment:

```python
# === FOR KAGGLE ENVIRONMENT ===
train_dir = "/kaggle/input/cardetection/car/train"

# === FOR LOCAL ENVIRONMENT ===
train_dir = "data/car/train"     # relative to project root
# or use absolute path:
train_dir = "/home/YOUR_USERNAME/projects/traffic-recognition/data/car/train"
```

### Step 2 — Load and Preprocess Data

```python
# Load images and bounding boxes from the dataset
X, y = load_images_and_bboxes(
    img_dir=os.path.join(train_dir, 'images'),
    label_dir=os.path.join(train_dir, 'labels'),
    img_size=(224, 224)
)

print(f"Images loaded: {X.shape}")    # e.g. (850, 224, 224, 3)
print(f"Labels loaded: {y.shape}")    # e.g. (850, 4)
print(f"Pixel range:   [{X.min():.2f}, {X.max():.2f}]")  # [0.00, 1.00]
```

### Step 3 — Visualize Ground Truth Bounding Boxes

```python
# Display three random images with their ground-truth bounding boxes
for _ in range(3):
    display_random_image_with_bbox(X, y)
```

The function:
1. Picks a random index from `[0, len(images) - 1]`
2. Converts the image from BGR to RGB
3. Rescales normalized coordinates to pixel values using the image's actual dimensions
4. Draws a green `cv2.rectangle` with thickness 2
5. Renders with `matplotlib.pyplot.imshow`

### Step 4 — Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples:   {len(X_train)}")
print(f"Test samples:       {len(X_test)}")
```

### Step 5 — Build and Compile the Model

```python
# Load ResNet50 with ImageNet weights, excluding the classification head
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all backbone layers — no gradient updates here
base_model.trainable = False

# Stack the custom regression head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(4, activation='sigmoid')      # [x_min, y_min, x_max, y_max]
])

# Compile with MSE loss for regression
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='mean_squared_error',
    metrics=['accuracy']
)

# Inspect the model
model.summary()
```

**Expected `model.summary()` output (abbreviated):**

```
Model: "sequential"
_________________________________________________________________
 Layer (type)               Output Shape          Param #   
=================================================================
 resnet50 (Functional)      (None, 7, 7, 2048)   23587712  
                                                             
 global_average_pooling2d   (None, 2048)          0         
                                                             
 dense (Dense)              (None, 1024)          2098176   
                                                             
 dense_1 (Dense)            (None, 4)             4100      
                                                             
=================================================================
Total params: 25,689,988
Trainable params: 2,102,276       ← Only the custom head trains
Non-trainable params: 23,587,712  ← ResNet50 backbone is frozen
_________________________________________________________________
```

### Step 6 — Train the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
```

**Expected training output:**

```
Epoch 1/10
27/27 [==============================] - 18s 580ms/step
  - loss: 0.0842 - accuracy: 0.0000e+00
  - val_loss: 0.0621 - val_accuracy: 0.0000e+00
Epoch 2/10
27/27 [==============================] - 15s 542ms/step
  - loss: 0.0534 - accuracy: 0.0000e+00
  - val_loss: 0.0489 - val_accuracy: 0.0000e+00
...
Epoch 10/10
27/27 [==============================] - 15s 538ms/step
  - loss: 0.0198 - accuracy: 0.0000e+00
  - val_loss: 0.0231 - val_accuracy: 0.0000e+00
```

> **Note on Accuracy:** For regression tasks, `accuracy` is not a meaningful metric (it will show `0.0`). Accuracy is suited for classification; MSE loss is the correct objective here. See the [Roadmap](#️-roadmap) for IoU implementation.

### Step 7 — Evaluate

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test MSE Loss:  {test_loss:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")   # Will be ~0.0; ignore this metric
```

### Step 8 — Run Inference on a New Image

```python
# Predict bounding box on any image path
predict_and_visualize(model, "path/to/your/image.jpg")
```

**What happens inside `predict_and_visualize`:**

```python
def predict_and_visualize(model, image_path):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape

    # Preprocess
    image_resized    = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_expanded   = np.expand_dims(image_normalized, axis=0)   # (1, 224, 224, 3)

    # Predict normalized bounding box
    predicted_bbox = model.predict(image_expanded)[0]              # (4,)
    x_min, y_min, x_max, y_max = predicted_bbox

    # Rescale to original pixel space
    x_min = max(0, int(x_min * original_width))
    y_min = max(0, int(y_min * original_height))
    x_max = min(original_width,  int(x_max * original_width))
    y_max = min(original_height, int(y_max * original_height))

    # Draw and display
    image_with_bbox = cv2.rectangle(image.copy(),
                                    (x_min, y_min),
                                    (x_max, y_max),
                                    (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
```

### Step 9 — Save the Trained Model

```python
# Keras format (recommended for TF 2.x)
model.save("traffic_sign_detector.keras")

# HDF5 format (legacy)
model.save("traffic_sign_detector.h5")

# TensorFlow SavedModel format (for serving / deployment)
model.save("saved_model/traffic_sign_detector")
```

### Step 10 — Load and Reuse

```python
from tensorflow.keras.models import load_model

model = load_model("traffic_sign_detector.keras")
predict_and_visualize(model, "test_image.jpg")
```

---

## 📈 Results

> ℹ️ The table below contains placeholder values. Fill these in after running your training on the actual dataset.

### Training Summary

| Metric | Epoch 1 | Epoch 5 | Epoch 10 |
|---|---|---|---|
| Train Loss (MSE) | `—` | `—` | `—` |
| Val Loss (MSE) | `—` | `—` | `—` |

### Final Evaluation

| Metric | Value |
|---|---|
| Test MSE Loss | `—` |
| Epochs Trained | 10 |
| Batch Size | 32 |
| Training Time | `—` |

### Visualizations

**Ground Truth Bounding Boxes (Training Set)**

> *(Add screenshot: `assets/sample_gt_bbox.png`)*

```
┌──────────────────────────────────────┐
│                                      │
│   ┌──────────────────┐               │
│   │   [Traffic Sign] │               │
│   └──────────────────┘               │
│                                      │
└──────────────────────────────────────┘
  Ground truth green bounding box
  rendered by display_random_image_with_bbox()
```

**Predicted Bounding Boxes (Test Set)**

> *(Add screenshot: `assets/sample_prediction.png`)*

```
┌──────────────────────────────────────┐
│                                      │
│     ┌────────────────────┐           │
│     │  Model Prediction  │           │
│     └────────────────────┘           │
│                                      │
└──────────────────────────────────────┘
  Predicted green bounding box
  rendered by predict_and_visualize()
```

### Metrics Interpretation

| Metric | Why It Matters | Target |
|---|---|---|
| **MSE Loss** | Measures average squared coordinate error — lower is better | As low as possible |
| **IoU** *(planned)* | Measures bounding box overlap with ground truth — higher is better | > 0.5 = good detection |
| **Accuracy** | Not meaningful for regression — ignore this output | N/A |

---

## 🔬 Technical Deep Dive

### Coordinate System Conventions

YOLO-format annotations normalize all coordinates relative to image dimensions:

```
x_min_normalized = x_min_pixels / image_width
y_min_normalized = y_min_pixels / image_height
x_max_normalized = x_max_pixels / image_width
y_max_normalized = y_max_pixels / image_height
```

This means the same label file works regardless of image resolution — the model learns in a scale-invariant coordinate space. At inference time, the inverse transform is applied:

```python
x_min_pixels = int(x_min_normalized * original_width)
y_min_pixels = int(y_min_normalized * original_height)
```

### Transfer Learning Strategy

This project uses **feature extraction** (not full fine-tuning):

```
Strategy 1 — Feature Extraction (used here):
  Freeze all ResNet50 layers → train only Dense head
  Pros: fast convergence, low risk of overfitting on small datasets
  Cons: backbone features may not be perfectly aligned to this domain

Strategy 2 — Fine-tuning (future enhancement):
  Unfreeze top N layers of ResNet50 → train both backbone tail + head
  Pros: domain-adapted features, potentially better localization
  Cons: slower, needs careful LR scheduling to avoid catastrophic forgetting
```

### Activation Function Choice

| Layer | Activation | Reason |
|---|---|---|
| Dense(1024) | ReLU | Prevents vanishing gradients; standard for hidden layers |
| Dense(4) | Sigmoid | Constrains output to [0,1]; matches normalized label space |

---

## 🛠️ Troubleshooting

### Installation Issues

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python
# If that fails on headless servers:
pip install opencv-python-headless
```

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
pip install tensorflow
# For GPU:
pip install tensorflow[and-cuda]
```

**`ImportError: cannot import name 'ResNet50' from 'keras'`**
```bash
# Ensure you're importing from tf.keras, not standalone keras:
from tensorflow.keras.applications import ResNet50
```

---

### Dataset Issues

**`FileNotFoundError` when loading images**

Check that `train_dir` points to the correct path and that `images/` and `labels/` subdirectories exist:

```python
import os
train_dir = "data/car/train"
print(os.path.exists(os.path.join(train_dir, 'images')))   # Should be True
print(os.path.exists(os.path.join(train_dir, 'labels')))   # Should be True
```

**`X` array is empty after `load_images_and_bboxes()`**

The function only loads `.jpg` and `.png` files. Verify your image extensions:

```bash
ls data/car/train/images/ | awk -F. '{print $NF}' | sort | uniq -c
```

**Kaggle API `401 Unauthorized`**

Regenerate your API token at [https://www.kaggle.com/settings](https://www.kaggle.com/settings) and replace `~/.kaggle/kaggle.json`.

---

### Training Issues

**Notebook kernel dies / crashes during training (Out of Memory)**

Reduce batch size:

```python
history = model.fit(X_train, y_train, epochs=10, batch_size=8, ...)  # Was 32
```

Or enable GPU memory growth to prevent TensorFlow from allocating all VRAM at once:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

**Training loss is `nan` from the first epoch**

This usually indicates a learning rate issue or malformed labels. Check:

```python
# Verify label values are in valid range
print(y.min(), y.max())    # Should be 0.0 and ~1.0
print(np.isnan(y).any())   # Should be False
```

**Accuracy stuck at 0.0 throughout training**

This is expected for regression tasks — `accuracy` is irrelevant here. Monitor `loss` and `val_loss` instead.

**`val_loss` is much higher than `loss` (overfitting)**

With the backbone frozen, this is less likely, but if it occurs:
- Reduce `Dense(1024)` to `Dense(512)` or add a `Dropout(0.3)` layer
- Add more training data or data augmentation

---

### Inference Issues

**`predict_and_visualize` throws `NoneType` error**

`cv2.imread()` returns `None` when the path is invalid or the file is unreadable:

```python
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
```

**Bounding box renders outside image boundaries**

This is handled by clamping in `predict_and_visualize`:

```python
x_min = max(0, x_min)
x_max = min(original_width, x_max)
y_min = max(0, y_min)
y_max = min(original_height, y_max)
```

Verify these lines are present in your copy of the function.

---

## 🗺️ Roadmap

This project currently implements **localization only**. The following enhancements are planned in priority order:

### Short Term

- [ ] **IoU Metric** — Implement Intersection over Union as a Keras custom callback for meaningful localization quality reporting
  ```python
  def iou(y_true, y_pred):
      # Compute intersection area
      inter_x1 = tf.maximum(y_true[:, 0], y_pred[:, 0])
      inter_y1 = tf.maximum(y_true[:, 1], y_pred[:, 1])
      inter_x2 = tf.minimum(y_true[:, 2], y_pred[:, 2])
      inter_y2 = tf.minimum(y_true[:, 3], y_pred[:, 3])
      intersection = tf.maximum(0.0, inter_x2 - inter_x1) * tf.maximum(0.0, inter_y2 - inter_y1)
      # Compute union area
      area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
      area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
      union = area_true + area_pred - intersection
      return tf.reduce_mean(intersection / (union + 1e-7))
  ```

- [ ] **Training History Visualization** — Plot `loss` vs `val_loss` curves using `matplotlib` from the `history` object

- [ ] **Data Augmentation** — Add `tf.keras.layers.RandomFlip`, `RandomBrightness`, `RandomContrast`, and `RandomZoom` for better generalization

- [ ] **Model Checkpointing** — Save the best epoch weights using `ModelCheckpoint` callback

### Medium Term

- [ ] **Classification Head** — Add a second output branch predicting traffic object class (stop sign, speed limit, traffic light, etc.), transforming this into a multi-task network:
  ```
  ResNet50 → GAP → Dense(1024) ┬→ Dense(4, sigmoid)        [bbox regression]
                                └→ Dense(num_classes, softmax) [classification]
  ```

- [ ] **Multi-object Detection** — Extend beyond single-object localization using anchor-based or anchor-free detection

- [ ] **Backbone Fine-tuning** — Unfreeze the top ResNet50 block (`conv5_block`) and train with a lower learning rate (1e-5) for domain adaptation

- [ ] **Model Export** — Export to TFLite (`.tflite`) for edge deployment and ONNX for cross-framework compatibility

### Long Term

- [ ] **YOLOv8 Baseline Comparison** — Benchmark against a YOLOv8n trained on the same dataset for speed/accuracy tradeoff analysis

- [ ] **Streamlit Web App** — Deploy the model as an interactive web demo with file upload and real-time bounding box visualization

- [ ] **Video Inference** — Extend `predict_and_visualize` to process video frames from `cv2.VideoCapture`

- [ ] **Real-time Webcam Demo** — Live inference pipeline using OpenCV's camera capture API

---

## 🤝 Contributing

Contributions are welcome and appreciated. Please follow the process below to maintain code quality and project coherence.

### How to Contribute

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CNN-based-Road-Traffic-Sign-and-Light-Recognition.git
   cd CNN-based-Road-Traffic-Sign-and-Light-Recognition
   ```
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
4. **Make your changes** with clean, documented code
5. **Test your changes** — run the full notebook end-to-end to verify nothing breaks
6. **Commit** using [Conventional Commits](#conventional-commits):
   ```bash
   git commit -m "feat: add IoU metric as Keras custom function"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
8. Open a **Pull Request** against `main` and describe your changes clearly

### Conventional Commits

All commits must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<optional scope>): <short description>
│          │                     │
│          │                     └── Imperative mood, no period, max 72 chars
│          └────────────────────── Optional: model, data, viz, eval, docs
└───────────────────────────────── One of the types below
```

| Type | When to Use |
|---|---|
| `feat` | A new feature or enhancement |
| `fix` | A bug fix |
| `docs` | Documentation changes only (README, comments, docstrings) |
| `refactor` | Code restructuring without behavior change |
| `perf` | Performance improvements (speed, memory) |
| `test` | Adding or updating tests or validation code |
| `chore` | Maintenance, dependency updates, config changes |
| `style` | Formatting, whitespace — no logic changes |

**Examples:**

```bash
git commit -m "feat(eval): implement IoU metric as custom Keras function"
git commit -m "fix(data): handle malformed label files with fewer than 5 tokens"
git commit -m "docs: add architecture ASCII diagram to README"
git commit -m "perf(train): enable GPU memory growth to prevent OOM crashes"
git commit -m "refactor(viz): extract bounding box rendering into standalone utility"
git commit -m "chore: add opencv-python to requirements.txt"
```

### Code Style Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code
- Add docstrings to all functions using Google style:
  ```python
  def load_images_and_bboxes(img_dir, label_dir, img_size=(224, 224)):
      """Load and preprocess images and bounding box annotations.

      Args:
          img_dir (str): Path to the directory containing image files.
          label_dir (str): Path to the directory containing YOLO-format label files.
          img_size (tuple): Target image dimensions (width, height). Defaults to (224, 224).

      Returns:
          tuple: (images, bboxes) where images is np.ndarray of shape (N, H, W, 3)
                 and bboxes is np.ndarray of shape (N, 4).
      """
  ```
- Keep notebook cells focused and atomic — one logical operation per cell
- Add a Markdown heading above each new notebook section

---

## 📚 Citation

If you use this project in your research, coursework, or derivative work, please cite:

```bibtex
@misc{wittyswayam2025cnntraffic,
  author       = {Swayam},
  title        = {CNN-based Road Traffic Sign and Light Recognition},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/wittyswayam/CNN-based-Road-Traffic-Sign-and-Light-Recognition}},
  note         = {Transfer learning with ResNet50 for bounding box regression on traffic sign datasets}
}
```

If your work builds on the foundational ResNet50 architecture, please also cite:

```bibtex
@inproceedings{he2016resnet,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016}
}
```

---

## 🙏 Acknowledgements

This project would not have been possible without the following open-source tools, datasets, and research:

- **[Keras Applications — ResNet50](https://keras.io/api/applications/resnet/)** — for providing a production-grade pretrained ResNet50 implementation with a simple, well-documented API
- **[TensorFlow / Keras Team](https://www.tensorflow.org/)** — for building and maintaining the deep learning framework used throughout
- **[Kaggle](https://www.kaggle.com/)** — for hosting the Car Detection dataset and providing free GPU-accelerated notebook environments
- **[OpenCV](https://opencv.org/)** — for the image loading, resizing, color conversion, and bounding box drawing utilities
- **[scikit-learn](https://scikit-learn.org/)** — for the clean, reliable `train_test_split` utility
- **[He et al. (2016)](https://arxiv.org/abs/1512.03385)** — for the original ResNet architecture paper that revolutionized deep learning
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — whose dataset annotation format is used as the label standard in this project
- The broader **computer vision and autonomous driving research community** for advancing open science in road scene understanding

---
