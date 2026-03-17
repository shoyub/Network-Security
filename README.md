````markdown
# Phishing Detection System

ML-powered web app that detects phishing URLs using hybrid scoring (ML model + heuristic patterns).

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8000/
```
````

## 3 Main Features

**01 Model Training** - Click "Initialize Training" (10-30 mins, one-time)

- Trains 4 ML algorithms: Random Forest, AdaBoost, Gradient Boosting, Logistic Regression
- Saves best model to `final_model/model.pkl`

**02 CSV Prediction** - Upload phishing dataset for batch analysis

- Predicts all rows instantly
- Returns results table

**03 URL Prediction** - Enter any URL for real-time analysis

- **ML Score**: Model confidence (0-100%)
- **Risk Score**: Pattern-based heuristics (0-100)
- Final verdict: 🚨 Phishing or ✅ Legitimate

## Key Files

| File                        | Purpose                         |
| --------------------------- | ------------------------------- |
| app.py                      | FastAPI backend (7 routes)      |
| `templates/index.html`      | Main UI with 3 sections         |
| `url_feature_extraction.py` | URL parser + heuristic scorer   |
| training_pipeline.py        | ML model training orchestration |
| `final_model/model.pkl`     | Pre-trained model               |
| `phisingData.csv`           | 11,000 training samples         |

## Tech Stack

FastAPI • scikit-learn • Pandas • MongoDB • Docker
