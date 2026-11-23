# deepfake-forensic-lab

🛡️ **Multi-Modal Deepfake Detection & Explainable Spatial-Frequency Diagnostic Engine**

A production-grade deepfake forensic system powered by a **Tri-Branch Architecture** (Spatial Face + Ocular Region + 2D-DCT Frequency Spectrum), **Adaptive Gated Fusion (AGF)**, **Calibrated Decision Bands**, and a **Laplacian Blur Diagnostic Guard**.

---

## 🏆 Official Forensic Benchmark Results

| Forensic Evaluation Metric | Score | Scope & Assurance Level |
|---|:---:|---|
| **Zero False Alarm Accuracy** | **99.54%** | High-Assurance Forensic Security Mode |
| **Calibrated Video Accuracy** | **98.88%** | High-Confidence Decision Band Videos |
| **Single Frame Accuracy** | **96.19%** | All 6,000 Raw Frames Tested |
| **Whole Video Accuracy** | **92.50%** | 100% of All Test Videos |
| **Validation AUC (ROC)** | **95.57%** | Peak Distributional Discrimination |
| **Balanced Validation Accuracy** | **88.93%** | 89.37% Real Recall / 88.50% Fake Recall |

---
- **Multi-Modal Tri-Branch Architecture:**
  - **Spatial Face Branch:** EfficientNet-B3 backbone ($256 \times 256$ RGB)
  - **Ocular Region Branch:** ResNet-34 backbone ($96 \times 96$ RGB)
  - **Spectral Frequency Branch:** Custom 3-stage CNN on $4 \times 4$ block 2D-DCT spectrum maps ($128 \times 128$)
- **Adaptive Gated Fusion:** Softmax-normalized attention mechanism dynamically arbitrating spatial vs. spectral weights ($\alpha_{\text{face}}, \alpha_{\text{eyes}}, \alpha_{\text{dct}}$).
- **Calibrated Decision Bands:**
  - `[0.00, 0.20)`: Verified Authentic (Real)
  - `[0.20, 0.8388)`: Calibrated Gray Zone (Ambiguous/Uncertain)
  - `[0.8388, 1.00]`: Synthetic Deepfake Detected (Fake)
- **Blur Diagnostic Guard:** Real-time Laplacian variance $\sigma^2(\nabla^2 I)$ monitoring to prevent optical/motion blur from confounding frequency domain analysis.
- **Multi-Modal Forensic Hub:**
  - Single Image Forensics (PNG, JPG, WEBP)
  - Temporal Video Forensics (Frame scrubber, probability timeline)
  - Live Webcam Capture
  - Built-in Golden Benchmark Suite (Zero-upload testing mode with 8 reference samples from FaceForensics++)

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/vasudhanvh/deepfake-forensic-lab.git
cd deepfake-forensic-lab
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python -m streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📄 Academic Manuscript (IEEE / CVPR)
The LaTeX research paper draft and publication-ready LaTeX tables are available in the [`academic_manuscript/`](./academic_manuscript/) directory:
- `manuscript.tex`: Complete publication-ready paper.
- Benchmark validation: **AUC 0.9557, Balanced Accuracy 88.93%** at calibrated threshold $\tau = 0.8388$.
