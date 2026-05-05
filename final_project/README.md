# Lifestyle Risk-Based Rate Recommendation

This Part 4 project adds a small end-to-end workflow:

1. Download and transform real data from the Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) and the Centers for Medicare & Medicaid Services (CMS) Marketplace Rate Public Use File (Rate PUF).
2. Load the Decision Tree model generated for Part 4.
3. Predict a lifestyle risk tier.
4. Calculate an insurance rate recommendation using a Centers for Medicare & Medicaid Services (CMS) benchmark base rate.
5. Write the result back to PostgreSQL.
6. Use SQLAlchemy ORM mappings for database inserts and updates.

## Install dependencies

From `c:\Users\craws\Documents\DMSProject4`:

```powershell
pip install -r final_project\app\requirements.txt
```

## Build real data sources

```powershell
python final_project\app\build_real_data_sources.py
```

This downloads official Centers for Disease Control and Prevention (CDC) BRFSS 2023 and Centers for Medicare & Medicaid Services (CMS) 2024 Marketplace Rate PUF files into `final_project\data_raw`, then writes clean project extracts into `final_project\model`. The script reads the large source files in chunks but uses every valid BRFSS row it can transform.

## Run without database writes

Build the real data sources and retrain the model first.

Use this for a quick screenshot of the analytics output. The app will ask questions interactively:

```powershell
python final_project\app\quote_risk_app.py --dry-run
```

## Run against PostgreSQL

Set your database connection string first:

```powershell
$env:DMS_DATABASE_URL = "postgresql://USER:PASSWORD@HOST:5432/DATABASE"
```

Then run:

```powershell
python final_project\app\quote_risk_app.py --skip-refresh-view
```

The app creates `QuoteRecommendation` if it does not already exist. The table name remains `QuoteRecommendation`, but the report should describe the business output as a rate recommendation.

## Retrain the model

```powershell
python final_project\app\retrain_model.py
```

The retraining script reads `final_project\model\brfss_lifestyle_risk_training.csv`.
The generated model file is written to `final_project\model\chronic_disease_risk_model.joblib`.

## Data Sources

- Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) 2023 annual survey data: used for lifestyle and health-risk model training.
- Centers for Medicare & Medicaid Services (CMS) Marketplace Rate Public Use File (Rate PUF) 2024: used to create age and tobacco benchmark monthly rates.

## ORM Use

The database workflow uses SQLAlchemy ORM models in `final_project\app\db_orm.py` for `Customer`, `ChronicDisease`, `RiskFactor`, `CustomerHealthProfile`, `HealthDataLakeRef`, and `QuoteRecommendation`.

## Topic framing

Use this wording in the report:

> Lifestyle Risk-Based Rate Recommendation

Use this insight question:

> Can lifestyle and behavioral risk factors help classify a customer into a low, medium, or high risk tier for quote review and rate recommendation?
