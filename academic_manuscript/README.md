# Academic Manuscript & Publication Materials (IEEE / CVPR Format)

This directory contains the IEEE / CVPR formatted publication draft and LaTeX tables for the multi-modal deepfake detection research paper.

## Manuscript Files
- **`manuscript.tex`**: Complete, self-contained, publication-ready LaTeX paper adhering to IEEE Transactions / CVPR formatting guidelines.

## Included LaTeX Tables & Methodologies
1. **Methodology Formulation**:
   - Explicit equations for Holistic Spatial Face Branch (EfficientNet-B3, $256\times256$).
   - Ocular Periocular Region Branch (ResNet-34, $96\times96$).
   - 2D-DCT Frequency Spectrum Branch (Custom SpecCNN, $128\times128$).
   - Adaptive Gated Fusion (AGF) bottleneck MLP, softmax attention formulation, and convex feature combination.
   - Classification head and Binary Cross-Entropy (BCE) objective with Exponential Moving Average (EMA).

2. **Table I**: Architectural parameter budget and input/output layer dimensions across all three branches and gating layers.
3. **Table II**: Empirical convergence and optimization trajectory across epochs 1 through 8 from `training_history.csv` on the FaceForensics++ benchmark.
4. **Table IV**: Comparative benchmark analysis contrasting individual single branches vs. uniform concatenation vs. Adaptive Gated Fusion vs. Calibrated Decision Band optimization ($\tau = 0.8388$).
5. **Table V**: Ablation study and Blur Diagnostic Guard robustness metrics under Gaussian blur ($\sigma \in \{1, 2\}$), motion blur ($k=9$), and lossy JPEG compression ($Q \in \{50, 30\}$).

## Compilation Instructions
To compile the document to PDF using standard TeX distributions (TeX Live, MacTeX, or MiKTeX):

```bash
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Or using Overleaf:
1. Create a New Project -> Upload Project.
2. Upload `manuscript.tex`.
3. Set Compiler to `pdfLaTeX` and click **Recompile**.
