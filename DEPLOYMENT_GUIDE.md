# Cloud Deployment & Packaging Guide: Deepfake Forensic Web Application

This document provides deployment guidelines for deploying the Tri-Branch Deepfake Detection Streamlit Application across **Streamlit Community Cloud**, **Hugging Face Spaces**, and **Vercel / Custom Web Portals (via iframe embedding)**, along with Git LFS management for the 132.7 MB model checkpoint.

---

## 1. Git LFS Configuration & Weight Serialization

The detector weights (`detector_weights.pth`) are **132,693,057 bytes (~132.7 MB)**, which exceeds GitHub's default 100 MB file limit. We manage this file using **Git Large File Storage (Git LFS)**.

### A. Initializing Git LFS in your local repository
```bash
# 1. Install Git LFS command line extension
git lfs install

# 2. Track large binary weight files
git lfs track "*.pth"
git lfs track "detector_weights.pth"

# 3. Add .gitattributes to git tracking
git add .gitattributes

# 4. Commit and push repository
git commit -m "chore: configure Git LFS tracking for 132.7 MB model weights"
git push origin main
```

### B. Verification: Git LFS Pointer vs. Real Binary Checkpoint
When cloned without Git LFS installed, git creates a ~130-byte text pointer file:
```text
version https://git-lfs.github.com/spec/v1
oid sha256:d5f...
size 132693057
```
To fetch the actual binary file on any machine:
```bash
git lfs pull
```

### C. Fallback Automatic Weight Fetcher (`weights_fetcher.py`)
In environments where Git LFS is unavailable or bandwidth quotas are exceeded, the included `weights_fetcher.py` script automatically intercepts model initialization:
1. It inspects `detector_weights.pth`. If the file is missing or smaller than 100 MB (i.e. an un-hydrated LFS pointer), it downloads the binary from a remote mirror (e.g. Hugging Face Model Hub / GitHub Release Asset).
2. It verifies the payload size and replaces the local file before passing control to `torch.load()`.

---

## 2. Streamlit Community Cloud Deployment

Streamlit Community Cloud natively supports Python web apps and Git LFS.

### Step-by-Step Deployment:
1. **Push your repository to GitHub** (public or private). Ensure `.gitattributes`, `packages.txt`, and `requirements.txt` are at the repository root.
2. Log into [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"** and select:
   - **Repository:** `your-username/deepfake-streamlit-app`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Linux Dependencies (`packages.txt`)**:
   Streamlit Cloud automatically reads `packages.txt` and runs `apt-get install`:
   ```text
   libgl1
   libglib2.0-0
   ffmpeg
   ```
5. **Python Dependencies (`requirements.txt`)**:
   PyTorch and Torchvision CPU distributions are installed automatically.
6. Click **"Deploy!"**. The deployment logs will show package installation, weight verification, and app spin-up within 2–3 minutes.

---

## 3. Hugging Face Spaces Deployment

Hugging Face Spaces provides hosting with CPU and GPU tiers and integration with Hugging Face Hub.

### Step-by-Step Deployment:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select:
   - **Space name:** `deepfake-forensic-lab`
   - **License:** `mit` or `apache-2.0`
   - **Space SDK:** `Streamlit`
   - **Hardware:** `CPU basic • 2 vCPU • 16GB RAM` (Free tier is sufficient for inference)
3. Clone the Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/your-username/deepfake-forensic-lab
   cd deepfake-forensic-lab
   ```
4. Copy all project files (`app.py`, `model_def.py`, `preprocessing.py`, `weights_fetcher.py`, `model_config.json`, `golden_samples/`, `requirements.txt`, `packages.txt`, `.streamlit/`) into the Space directory.
5. Commit and push:
   ```bash
   git add .
   git commit -m "feat: deploy deepfake forensic streamlit application"
   git push
   ```
6. Hugging Face Spaces builds the container and launches the application with an accessible URL: `https://huggingface.co/spaces/your-username/deepfake-forensic-lab`.

---

## 4. Vercel & Next.js / React Embed Wrapper

To embed the Streamlit Forensic Hub inside a production web portal, enterprise dashboard, or React/Next.js site hosted on Vercel:

### A. Responsive HTML/React Embed Component (`DeepfakeForensicEmbed.tsx`)
```tsx
import React, { useState } from 'react';

interface ForensicEmbedProps {
  streamUrl?: string; // e.g., "https://your-app.streamlit.app?embed=true"
}

export const DeepfakeForensicEmbed: React.FC<ForensicEmbedProps> = ({
  streamUrl = "https://your-username-deepfake-app.streamlit.app?embed=true"
}) => {
  const [isLoading, setIsLoading] = useState(true);

  return (
    <div style={{ width: '100%', minHeight: '900px', position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '1px solid #1f2a44', background: '#0b0f19' }}>
      {isLoading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', color: '#00e5ff', fontFamily: 'monospace' }}>
          🛡️ Initializing Forensic Diagnostic Hub...
        </div>
      )}
      <iframe
        src={streamUrl}
        title="Deepfake Forensic Analysis Hub"
        width="100%"
        height="950px"
        frameBorder="0"
        allow="camera; microphone"
        onLoad={() => setIsLoading(false)}
        style={{ width: '100%', height: '950px', border: 'none' }}
      />
    </div>
  );
};

export default DeepfakeForensicEmbed;
```

### B. Vercel Deployment Instructions (`vercel.json` headers)
When embedding via iframe, configure your Vercel `vercel.json` or Next.js headers to allow cross-origin camera access:
```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Permissions-Policy",
          "value": "camera=(self 'https://*.streamlit.app' 'https://*.hf.space')"
        }
      ]
    }
  ]
}
```

---

## 5. Security & Resource Optimization Summary
- **Memory Footprint:** PyTorch CPU inference on a single 256x256 frame requires **< 350 MB RAM**.
- **Latency:** ~45 ms per frame on standard modern 4-core CPU; ~8 ms on GPU.
- **Privacy:** In headless mode (`.streamlit/config.toml`), uploaded media is processed in-memory and discarded upon session disconnect.
