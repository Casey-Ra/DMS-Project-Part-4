# Part 4 Proof Steps

GitHub repository:

https://github.com/Casey-Ra/DMS-Project-Part-4

## Project Focus

Lifestyle Risk-Based Insurance Quote Support

Insight question:

Can lifestyle and behavioral risk factors help classify a customer into a low, medium, or high risk tier for quote review and rate recommendation?

## What Was Built

- `final_project/app/quote_risk_app.py`
  - loads the trained Decision Tree model
  - accepts customer lifestyle inputs
  - predicts a risk tier
  - calculates a sample monthly rate recommendation
  - can write the result to PostgreSQL

- `final_project/app/retrain_model.py`
  - reloads the CSV training dataset
  - retrains the Decision Tree model
  - saves the updated `.joblib` model artifact

- `final_project/database/ProjectPart4_QuoteSupportSchema.sql`
  - adds a `QuoteRecommendation` table for the rate recommendation result

- `final_project/report/final_project_report_outline.md`
  - gives the final report structure and wording

## Proof Command 1: Run the App Without Database Writes

Command:

```powershell
python final_project\app\quote_risk_app.py --dry-run --first-name Casey --last-name Demo --age 45 --exercise-level 2 --unhealthy-eating-level 3 --smoking-drinking 1 --heredity 1 --living-standard 3 --base-monthly-rate 220
```

Observed output:

```text
Lifestyle Risk-Based Quote Support Result
-----------------------------------------
Customer: Casey Demo
Model high-risk prediction: 1
High-risk probability: 1.00
Risk tier: High
Primary risk factor: Smoking/Drinking
Base monthly rate: $220.00
Rate adjustment factor: 1.25
Recommended monthly rate: $275.00
Recommendation reason: High lifestyle risk tier based on model output, 1.00 high-risk probability, and 4 risk flags.

Dry run complete. No database rows were written.
```

Screenshot to capture:

- Terminal showing the command and the quote-support result.

## Proof Command 2: Retrain the ML Model

Command:

```powershell
python final_project\app\retrain_model.py
```

Observed output:

```text
Lifestyle Risk Model Retraining Complete
----------------------------------------
Training rows: 20
Feature columns: age, exercise_level, unhealthy_eating_level, smoking_drinking, heredity, living_standard
Accuracy: 1.00
Saved retrained model: C:\Users\craws\Documents\DMSProject4\final_project\model\chronic_disease_risk_model.joblib
```

Screenshot to capture:

- Terminal showing the retraining command and successful model save.

## Optional Proof Command 3: Run Against PostgreSQL

Set the connection string:

```powershell
$env:DMS_DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

Run the workflow:

```powershell
python final_project\app\quote_risk_app.py --first-name Casey --last-name Demo --age 45 --exercise-level 2 --unhealthy-eating-level 3 --smoking-drinking 1 --heredity 1 --living-standard 3 --base-monthly-rate 220
```

Screenshot to capture:

- Terminal showing `Database update complete`.
- SQL query showing the inserted customer, health profile, or quote recommendation.

## Suggested SQL Verification

```sql
SELECT *
FROM "QuoteRecommendation"
ORDER BY "QuoteRecommendationID" DESC
LIMIT 5;
```

```sql
SELECT *
FROM "CustomerHealthRiskSummary"
ORDER BY "HealthProfileID" DESC
LIMIT 5;
```
