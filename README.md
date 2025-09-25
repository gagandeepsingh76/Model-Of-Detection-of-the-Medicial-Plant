# 🌱 Plant Image Classification System

This project is a **deep learning-based system** for classifying **medicinal and edible plants** from images. It implements four CNN architectures and provides functionality for both **training** and **prediction**.

---

## 🔗 Models Access

The system trains and saves four models:

| Model | Filename | Description |
|-------|---------|-------------|
| Custom CNN | `m1.h5` | Simple CNN with Conv2D, MaxPooling2D, and Dense layers. Fast training, suitable for smaller datasets. |
| MobileNetV2 | `m2.h5` | Lightweight model optimized for mobile devices. Good balance of speed and accuracy. |
| EfficientNetB0 | `m3.h5` | State-of-the-art architecture with excellent accuracy at reasonable computational cost. |
| ResNet50 | `m4.h5` | Deep residual network with strong performance on complex classification tasks. |

---

## 🛠️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt

