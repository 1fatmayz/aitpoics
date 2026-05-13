# CSAI415 Deliverable 1 – Streaming Learner & AutoML

**Students:** Fatma Abdulla, Rand Abu Baker, Shaima Abu Serdaneh
**Course:** CSAI415  
**Deliverable:** D1 – Week 5 (15%)

## Project Overview
This project implements the required Deliverable 1 system for Track A: **Supervised auto-tuned kNN retriever**.

The system uses an AutoML approach to optimize retrieval performance by tuning:
- Number of neighbors (k)
- Distance metric
- SVD dimensionality
- Feature normalization
- Hybrid retrieval weighting

A streaming learning component is also included using River, with prequential evaluation and ADWIN drift detection for adaptive performance monitoring.

## Included Files
- `D1_Report.pdf` → Final report (metrics, evaluation, analysis)
- `D1_Project.ipynb` → Runnable notebook containing full implementation
- `run_card.json` → Winning configuration and final performance metrics

## How to Run
1. Open the notebook in Google Colab or Jupyter Notebook.
2. Install required libraries if needed:
```bash
pip install optuna river scikit-learn pandas numpy matplotlib rank-bm25