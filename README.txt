DEEPFAKE DETECTOR - DEPLOYMENT BUNDLE

Files
  detector_weights.pth   EMA weights (tensors only; torch.load(..., weights_only=True))
  model_config.json      threshold, architecture, input spec, training metadata
  model_def.py           standalone model + load_detector() + predict()
  training_history.csv   per-epoch metrics from training
  golden_samples/        a few val samples (face/eyes/dct) + expected probabilities

Quick start
  from model_def import load_detector, predict
  model, cfg = load_detector("path/to/this/folder", device="cpu")
  probs, gates = predict(model, face, eyes, dct)
  is_fake = probs >= cfg["threshold"]

Inputs (see model_config.json)
  face  (N,3,256,256) float32 in [0,1]
  eyes  (N,3,96,96)   float32 in [0,1]
  dct   (N,1,128,128) float32, same values as the dataset .npy files

IMPORTANT: this bundle does NOT include the face-detection / eye-crop / DCT
preprocessing that built the dataset. An app must reproduce that preprocessing
exactly, or predictions will be unreliable. Use golden_samples/ to check that
model loading works independently of preprocessing.
