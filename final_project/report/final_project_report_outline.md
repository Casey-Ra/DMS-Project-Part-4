Part 4 Report Draft

2. Business Use Cases, Process Models, and Application Design

2.1 Business Use Case

    The selected business use case is a lifestyle risk-based insurance rate recommendation workflow. The application allows a user to enter customer lifestyle information or load a free-text lifestyle note, run a trained machine learning model, calculate a rate recommendation, and save the result to the database.

    The application is designed as decision support. It does not automatically approve or deny insurance coverage. A human user reviews the risk tier, rate adjustment, and recommendation before using the result in a business decision.

    The main business question is: Can lifestyle and behavioral risk factors help classify a customer into a low, medium, or high risk tier for insurance rate recommendation?

2.2 Actors

    The main actors are the quote review user, the customer, the data pipeline, and the database. The quote review user enters or selects customer information. The customer is the person being evaluated. The data pipeline prepares external and unstructured data. The database stores the workflow result.

2.3 Business Use Case Model

Insert Figure 1 here.

Figure 1. Business Use Case Model

Image file: final_project/report/figures/figure_1_business_use_case.svg

    Figure 1 shows the business workflow from customer input to human review. The user enters information, the application creates model-ready features, the model predicts a risk tier, the application calculates a rate recommendation, and the result is saved to the database.

2.4 Process Model

Insert Figure 2 here.

Figure 2. Process Model

Image file: final_project/report/figures/figure_2_process_model.svg

    Figure 2 shows the full process. The first stage prepares the source data. The second stage retrains and saves the machine learning model. The third stage runs the application, calculates the rate recommendation, and saves the result to the cloud database.

2.5 Application Design

    The application is built in Python. The main files are build_real_data_sources.py, parse_unstructured_notes.py, retrain_model.py, quote_risk_app.py, and db_orm.py.

    The project uses real public-use data from the Centers for Disease Control and Prevention (CDC) and the Centers for Medicare & Medicaid Services (CMS). CDC data is used for lifestyle and health-risk model training. CMS data is used for the base rate benchmark. A small synthetic free-text notes file is used to demonstrate unstructured data processing.

2.6 Application Architecture

Insert Figure 3 here.

Figure 3. Application Architecture Model

Image file: final_project/report/figures/figure_3_application_architecture.svg

    Figure 3 shows how the application components work together. The user enters data manually or selects a note ID. The application converts the input into model features, loads the trained model file, predicts a risk tier, looks up a base rate, calculates the recommendation, and saves the result through the database layer.

    The rate rule is intentionally simple: low risk uses the base rate, medium risk uses the base rate multiplied by 1.10, and high risk uses the base rate multiplied by 1.25.

5. Application Documentation, End-to-End Integration, Screenshots, and Query Optimization

5.1 End-to-End Summary

    The solution is end-to-end because it connects data preparation, unstructured note parsing, model retraining, model prediction, rate recommendation, and database updates.

    The workflow is: build the CDC and CMS extracts, parse unstructured lifestyle notes, retrain the model, run the application, predict the risk tier, calculate the rate recommendation, save the result to the database, and verify the saved rows with SQL queries.

5.2 Data-Driven Module

    The main data-driven module is quote_risk_app.py. It loads customer input, optionally loads note-based features, runs the trained decision tree model, calculates the recommended monthly rate, and saves the result.

    The model uses age, tobacco use, obesity, physical inactivity, alcohol risk, diabetes, and general health as input features. The output is a low, medium, or high risk tier.

5.3 Data Pipeline and Retraining

    The data pipeline is handled by build_real_data_sources.py and parse_unstructured_notes.py. The retraining process is handled by retrain_model.py. If source data changes, the data scripts can be rerun and the model can be retrained without redesigning the application.

    Main commands:

python final_project\app\build_real_data_sources.py --force-download

python final_project\app\parse_unstructured_notes.py

python final_project\app\retrain_model.py

python final_project\app\quote_risk_app.py --note-id NOTE001

5.4 Database Integration and ORM

    The application uses SQLAlchemy ORM to save records to the database. The ORM classes are defined in db_orm.py. The workflow saves customer, health profile, data source reference, and quote recommendation records.

    Using ORM supports the extra-credit requirement and keeps the application code cleaner than writing every insert manually. SQLAlchemy also parameterizes database operations, which helps avoid unsafe SQL string construction.

5.5 Query Optimization

    The database uses indexes on common lookup columns such as customer ID, health profile ID, disease ID, risk factor ID, and quote recommendation references. These indexes support faster searches for customer profiles, source references, and saved recommendations.

    The database also uses a materialized summary view for customer health risk reporting. This reduces repeated join work when reviewing customer risk information.

5.6 Screenshots to Include

    Include screenshots showing the data source build script, note parsing script, model retraining script, application run, saved database result, and GitHub repository. A useful database verification screenshot is a query against QuoteRecommendation ordered by the newest records.

Example verification query:

SELECT *
FROM "QuoteRecommendation"
ORDER BY "QuoteRecommendationID" DESC
LIMIT 5;

6. End-to-End Reference Architecture and Governance

6.1 Reference Architecture Overview

    The reference architecture connects the business process, Python application, data pipeline, machine learning model, and cloud database. It is organized across business, application, data and knowledge, and infrastructure domains.

6.2 Organizing Framework

Insert Figure 4 here.

Figure 4. Reference Architecture Organizing Framework

Image file: final_project/report/figures/figure_4_reference_architecture.svg

    Figure 4 shows the main architecture domains. The business domain defines the rate recommendation use case. The application domain contains the Python app and database mapping. The data and knowledge domain contains the source data, notes, model, and metadata. The infrastructure domain contains the cloud database, GitHub repository, and local runtime. Governance applies across all of these areas.

6.3 Data-to-Decision Flow

Insert Figure 5 here.

Figure 5. Data-to-Decision Flow

Image file: final_project/report/figures/figure_5_dikw_flow.svg

    Figure 5 shows how the solution moves from raw data to business action. Raw data becomes cleaned information, the trained model produces knowledge in the form of a risk tier, and the final rate recommendation is reviewed and stored in the database.

6.4 Governance

    Governance is important because the project uses health-related lifestyle data and machine learning. The model should not be treated as an automatic final decision-maker. It should support human review.

    The main governance controls are data lineage, data quality checks, credential protection, privacy protection, and human review. Source references are stored in HealthDataLakeRef so the result can be traced back to CDC, CMS, and note-based inputs.

    Fairness and transparency are also important. The application shows the risk tier, model probability, base rate source, adjustment factor, and recommendation reason. This helps the user understand why the recommendation was produced.

6.5 Security and Data Protection

    Database credentials are supplied through environment variables and are not stored in the source code. The unstructured notes are synthetic, so the project does not expose real patient notes or protected health information.

    In a production system, additional controls would be needed, including encryption, role-based access control, audit logging, backups, and formal model monitoring.

6.6 Summary

    This Part 4 solution implements a complete data-driven workflow. It prepares public-use source data, parses unstructured lifestyle notes, retrains a model, calculates a rate recommendation, and saves the result to the database. The solution also includes ORM-based database integration, query optimization, architecture documentation, and governance controls.
