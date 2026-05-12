# Lifestyle Risk-Based Insurance Rate Recommendation

A machine learning workflow that predicts insurance risk tiers and recommends rates based on lifestyle factors. The app combines structured public health and insurance benchmark data with parsed lifestyle notes to produce a repeatable risk assessment pipeline. It includes scripts for building data sources, parsing unstructured notes, retraining the model, and running quote recommendations with or without a database connection.

## Setup

**Windows (PowerShell) or Mac:**
```powershell
pip install -r final_project\app\requirements.txt
```

## Quick Start

**Windows (PowerShell) or Mac:**
```powershell
python final_project\app\build_real_data_sources.py
python final_project\app\parse_unstructured_notes.py
python final_project\app\retrain_model.py

python final_project\app\quote_risk_app.py --dry-run
python final_project\app\quote_risk_app.py --dry-run --note-id NOTE001
```

## Run with Database

**Windows (PowerShell):**
```powershell
$env:DMS_DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
python final_project\app\quote_risk_app.py --skip-refresh-view
python final_project\app\quote_risk_app.py --skip-refresh-view --note-id NOTE001
```

**Mac/Linux (Bash):**
```bash
export DMS_DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DATABASE"
python final_project/app/quote_risk_app.py --skip-refresh-view
python final_project/app/quote_risk_app.py --skip-refresh-view --note-id NOTE001
```

## Data Sources

- CDC BRFSS 2023: Lifestyle training data
- CMS Rate PUF 2024: Insurance rate benchmarks
- Patient lifestyle notes: Unstructured text parsing demo
