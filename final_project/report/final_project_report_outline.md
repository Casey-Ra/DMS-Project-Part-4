# Final Project Report Outline

## 1. Project Focus

This project focuses on lifestyle risk-based insurance quote support. The goal is to use lifestyle and chronic-disease risk factors to classify a customer into a low, medium, or high risk tier, then recommend a sample rate adjustment for underwriting review.

## 2. Business Use Case

The workflow begins when a user enters customer lifestyle information. The Python application runs the trained machine learning model, calculates a risk tier, applies a sample rate adjustment, writes the insight back to the OLTP/ODS database, and displays the quote support result.

## 3. Data-Driven Module

The analytics module uses a Decision Tree classifier trained from lifestyle features:

- age
- exercise level
- unhealthy eating level
- smoking/drinking
- heredity
- living standard

The model output supports underwriting review. It does not automatically approve, deny, or price a real insurance policy.

## 4. Database Integration

The application writes results into:

- `Customer`
- `CustomerHealthProfile`
- `HealthDataLakeRef`
- `QuoteRecommendation`

The existing `CustomerHealthRiskSummary` materialized view can be refreshed after new results are inserted.

## 5. Data Pipeline and Retraining

For this class implementation, the CSV file represents the structured dataset extracted from a larger health-behavior data pipeline. The retraining script reloads the CSV, trains the Decision Tree model again, and saves an updated `chronic_disease_risk_model.joblib` artifact.

## 6. Query Optimization

The database design uses indexes on foreign keys and summary lookup columns. The app uses parameterized SQL queries to avoid unsafe SQL string construction. The materialized view supports faster reporting queries by avoiding repeated joins across customer, health profile, disease, risk factor, and data lake reference tables.

## 7. Governance

The model should be used as a decision-support tool only. The output should be reviewed by a human underwriter before any customer-facing insurance decision. The system should be monitored for bias, fairness, transparency, and explainability because lifestyle and health variables can affect customers unevenly.

## 8. Screenshots to Add

- Terminal app run showing risk tier and recommended monthly rate.
- Database query showing inserted customer health profile or quote recommendation.
- Retraining script output showing the model was rebuilt.
- Azure PostgreSQL screenshot or SQL query result.
