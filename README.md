# Derm Nexus AI

An academic deep-learning project for seven-class skin-lesion image classification, with a PyTorch training pipeline, Flask inference API, and Next.js interface.

> **Important:** This repository is an educational prototype, not a medical device or diagnostic service. Predictions can be wrong and must not replace evaluation by a qualified clinician.

## What is implemented

- Dataset loading and augmentation utilities
- EfficientNet-B3 classifier adapted for seven HAM10000-style classes
- Training and evaluation scripts
- Flask image-inference endpoint returning the top three class probabilities
- Next.js interface for uploading an image and reviewing model output
- Optional OpenAI-backed explanatory chat with a safe fallback when no key is configured

## Evidence and current limitations

The committed training history records a peak validation accuracy of approximately **55.3%** and a final recorded validation accuracy of approximately **53.9%**. Those results are suitable for coursework and iteration, not clinical use.

The interface's highlighted image regions are presentation overlays derived from class predictions; they are **not** produced by a lesion-localization or segmentation model. The project should not be described as locating suspicious tissue.

## Repository layout

```text
Ann Project/
├── src/                 # dataset, model, training, evaluation
├── models/              # committed training history
├── app/                 # Flask inference service
├── frontend/            # Next.js interface and chat route
├── requirements.txt     # Python dependencies
├── render.yaml          # backend deployment configuration
└── package.json         # workspace launcher scripts
```

## Run locally

From `Ann Project/`:

```bash
npm run install:all
```

Create `frontend/.env.local` only if you want the optional explanatory chat:

```bash
OPENAI_API_KEY=your_key
```

Then start the workspace:

```bash
npm run dev
```

The interface is available at [http://localhost:3000](http://localhost:3000).

## Model classes

The classifier covers actinic keratoses, basal cell carcinoma, benign keratosis, dermatofibroma, melanoma, melanocytic nevi, and vascular lesions.

## Responsible-use boundary

- Do not use the output to diagnose, rule out, or treat a condition.
- Do not present confidence scores as calibrated medical risk.
- Seek professional review for concerning or changing skin lesions.
- A production research path would require stronger validation, calibration, subgroup analysis, clinical review, privacy controls, and regulatory assessment.
