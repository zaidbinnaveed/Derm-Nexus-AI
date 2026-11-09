# Derm Nexus AI

**A seven-class skin-lesion image classifier with an end-to-end pipeline: PyTorch training, a Flask inference API, and a Next.js interface.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?logo=nextdotjs&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Not a medical device.** This is an academic prototype. Predictions can be wrong and must never replace evaluation by a qualified clinician.

## Overview

Derm Nexus fine-tunes an ImageNet-pretrained EfficientNet-B3 on HAM10000-style dermoscopy images across seven classes, serves the model behind a Flask endpoint that returns the top-3 class probabilities, and presents the result in a Next.js interface with an optional explanatory chat.

**Classes:** actinic keratoses, basal cell carcinoma, benign keratosis, dermatofibroma, melanoma, melanocytic nevi, vascular lesions.

## Results

| Metric | Value |
| --- | --- |
| Peak validation accuracy | **55.3%** |
| Final validation accuracy | 53.9% |
| Epochs recorded | 21 |

Source: [`models/training_history.json`](models/training_history.json). These are coursework-level results on an imbalanced dataset; accuracy alone is not an adequate metric here (see *Limitations*).

## Training approach

- **Backbone:** EfficientNet-B3 (torchvision, ImageNet weights) with a new two-layer classification head.
- **Two-phase fine-tuning:** 10 epochs with the backbone frozen (lr 1e-3), then 15 epochs unfrozen with cosine-annealed learning rate.
- **Augmentation:** Albumentations pipeline for geometric and color variation.

## Getting started

**Prerequisites:** Python 3.11, Node.js 18+.

```bash
git clone https://github.com/zaidbinnaveed/Derm-Nexus-AI.git
cd Derm-Nexus-AI

pip install -r requirements.txt      # inference API and training code
npm run install:all                  # frontend dependencies
npm run dev                          # http://localhost:3000
```

Start the inference API in another terminal:

```bash
python app/app.py
```

On first start the API downloads the trained weights. The optional chat needs `OPENAI_API_KEY` in `frontend/.env.local` (see `frontend/.env.example`); without it the app shows a safe fallback message.

To retrain:

```bash
python src/train.py
python src/evaluate.py
```

## Project structure

```text
src/                  dataset, model, training, and evaluation code
models/               committed training history
app/app.py            Flask inference API (top-3 probabilities)
frontend/             Next.js interface and chat route
scripts/              dev launcher
docs/                 project report and frontend notes
render.yaml           API deployment config (Render)
```

## Limitations

- Validation accuracy is modest; per-class precision and recall, calibration, and subgroup analysis (e.g., across skin tones) have not been evaluated.
- The highlighted regions in the interface are presentation overlays derived from the predicted class. **They are not produced by a localization or segmentation model.**
- Confidence scores are softmax outputs, not calibrated clinical risk.

## License

[MIT](LICENSE) © Zaid Bin Naveed
