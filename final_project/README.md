# Lifestyle Risk-Based Insurance Quote Support

This Part 4 starter adds a small end-to-end workflow for the final project:

1. Collect customer lifestyle risk inputs.
2. Load the Decision Tree model generated for Part 4.
3. Predict a lifestyle risk tier.
4. Calculate a sample insurance rate recommendation.
5. Write the result back to PostgreSQL.

## Install dependencies

From `c:\Users\craws\Documents\DMSProject4`:

```powershell
pip install -r final_project\app\requirements.txt
```

## Run without database writes

If the model file is not present yet, run the retraining command first.

Use this for a quick screenshot of the analytics output:

```powershell
python final_project\app\quote_risk_app.py --dry-run --first-name Casey --last-name Demo --age 45 --exercise-level 2 --unhealthy-eating-level 3 --smoking-drinking 1 --heredity 1 --living-standard 3 --base-monthly-rate 220
```

## Run against PostgreSQL

Set your database connection string first:

```powershell
$env:DMS_DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

Then run:

```powershell
python final_project\app\quote_risk_app.py --first-name Casey --last-name Demo --age 45 --exercise-level 2 --unhealthy-eating-level 3 --smoking-drinking 1 --heredity 1 --living-standard 3 --base-monthly-rate 220
```

The app creates `QuoteRecommendation` if it does not already exist.

## Retrain the model

```powershell
python final_project\app\retrain_model.py
```

The retraining script reads `final_project\model\chronic_disease_sample_data.csv`.
The generated model file is written to `final_project\model\chronic_disease_risk_model.joblib`.

## Topic framing

Use this wording in the report:

> Lifestyle Risk-Based Insurance Quote Support

Use this insight question:

> Can lifestyle and behavioral risk factors help classify a customer into a low, medium, or high risk tier for quote review and rate recommendation?
