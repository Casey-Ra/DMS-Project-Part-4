from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import pandas as pd

try:
    import psycopg2
except ImportError:  # pragma: no cover - handled at runtime for dry runs
    psycopg2 = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    PART4_ROOT
    / "model"
    / "chronic_disease_risk_model.joblib"
)
DEFAULT_DATASET_URI = "local://final_project/model/chronic_disease_sample_data.csv"

FEATURE_COLUMNS = [
    "age",
    "exercise_level",
    "unhealthy_eating_level",
    "smoking_drinking",
    "heredity",
    "living_standard",
]

RATE_ADJUSTMENTS = {
    "Low": Decimal("1.00"),
    "Medium": Decimal("1.10"),
    "High": Decimal("1.25"),
}

RISK_SCORES = {
    "Low": Decimal("35.00"),
    "Medium": Decimal("55.00"),
    "High": Decimal("75.00"),
}


@dataclass
class CustomerInput:
    first_name: str
    last_name: str
    age: int
    exercise_level: int
    unhealthy_eating_level: int
    smoking_drinking: int
    heredity: int
    living_standard: int
    base_monthly_rate: Decimal


@dataclass
class RiskResult:
    model_prediction: int
    risk_probability: float
    risk_tier: str
    risk_score: Decimal
    primary_risk_factor: str
    rate_adjustment_factor: Decimal
    recommended_monthly_rate: Decimal
    recommendation_reason: str


def decimal_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def prompt_if_missing(value: object | None, prompt: str, cast=str):
    if value is not None:
        return value
    raw = input(prompt).strip()
    return cast(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lifestyle risk-based insurance quote support demo."
    )
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument("--age", type=int)
    parser.add_argument("--exercise-level", type=int, choices=range(1, 5))
    parser.add_argument("--unhealthy-eating-level", type=int, choices=range(1, 5))
    parser.add_argument("--smoking-drinking", type=int, choices=(0, 1))
    parser.add_argument("--heredity", type=int, choices=(0, 1))
    parser.add_argument("--living-standard", type=int, choices=range(1, 5))
    parser.add_argument("--base-monthly-rate", type=Decimal)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DMS_DATABASE_URL") or os.getenv("PG_URI"),
        help="PostgreSQL connection URL. Defaults to DMS_DATABASE_URL or PG_URI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the ML and rate recommendation without writing to the database.",
    )
    parser.add_argument(
        "--skip-refresh-view",
        action="store_true",
        help="Skip refreshing CustomerHealthRiskSummary after inserting records.",
    )
    return parser.parse_args()


def read_customer_input(args: argparse.Namespace) -> CustomerInput:
    return CustomerInput(
        first_name=prompt_if_missing(args.first_name, "First name: "),
        last_name=prompt_if_missing(args.last_name, "Last name: "),
        age=prompt_if_missing(args.age, "Age: ", int),
        exercise_level=prompt_if_missing(
            args.exercise_level, "Exercise level (1=low, 4=high): ", int
        ),
        unhealthy_eating_level=prompt_if_missing(
            args.unhealthy_eating_level,
            "Unhealthy eating level (1=low, 4=high): ",
            int,
        ),
        smoking_drinking=prompt_if_missing(
            args.smoking_drinking, "Smoking/drinking? (0=no, 1=yes): ", int
        ),
        heredity=prompt_if_missing(args.heredity, "Heredity risk? (0=no, 1=yes): ", int),
        living_standard=prompt_if_missing(
            args.living_standard, "Living standard (1=low, 4=high): ", int
        ),
        base_monthly_rate=prompt_if_missing(
            args.base_monthly_rate, "Base monthly rate: ", Decimal
        ),
    )


def validate_customer_input(customer: CustomerInput) -> None:
    if customer.age < 0 or customer.age > 120:
        raise ValueError("Age must be between 0 and 120.")
    for label, value in [
        ("exercise level", customer.exercise_level),
        ("unhealthy eating level", customer.unhealthy_eating_level),
        ("living standard", customer.living_standard),
    ]:
        if value < 1 or value > 4:
            raise ValueError(f"{label.title()} must be from 1 to 4.")
    for label, value in [
        ("smoking/drinking", customer.smoking_drinking),
        ("heredity", customer.heredity),
    ]:
        if value not in (0, 1):
            raise ValueError(f"{label.title()} must be 0 or 1.")
    if customer.base_monthly_rate <= 0:
        raise ValueError("Base monthly rate must be greater than zero.")


def feature_frame(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": customer.age,
                "exercise_level": customer.exercise_level,
                "unhealthy_eating_level": customer.unhealthy_eating_level,
                "smoking_drinking": customer.smoking_drinking,
                "heredity": customer.heredity,
                "living_standard": customer.living_standard,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def identify_primary_risk_factor(customer: CustomerInput) -> str:
    if customer.smoking_drinking:
        return "Smoking/Drinking"
    if customer.unhealthy_eating_level >= 3:
        return "Unhealthy Eating"
    if customer.exercise_level <= 2:
        return "Low Exercise"
    if customer.heredity:
        return "Heredity"
    if customer.living_standard <= 2:
        return "Living Standard"
    if customer.age >= 50:
        return "Age"
    return "General Lifestyle"


def lifestyle_flag_count(customer: CustomerInput) -> int:
    return sum(
        [
            customer.age >= 50,
            customer.exercise_level <= 2,
            customer.unhealthy_eating_level >= 3,
            customer.smoking_drinking == 1,
            customer.heredity == 1,
            customer.living_standard <= 2,
        ]
    )


def score_risk(model, customer: CustomerInput) -> RiskResult:
    features = feature_frame(customer)
    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        risk_probability = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)
    else:
        risk_probability = float(prediction)

    flags = lifestyle_flag_count(customer)
    if prediction == 1 or risk_probability >= 0.67 or flags >= 4:
        tier = "High"
    elif risk_probability >= 0.35 or flags >= 2:
        tier = "Medium"
    else:
        tier = "Low"

    adjustment = RATE_ADJUSTMENTS[tier]
    recommended_rate = decimal_money(customer.base_monthly_rate * adjustment)
    primary_factor = identify_primary_risk_factor(customer)
    reason = (
        f"{tier} lifestyle risk tier based on model output, "
        f"{risk_probability:.2f} high-risk probability, and {flags} risk flags."
    )

    return RiskResult(
        model_prediction=prediction,
        risk_probability=risk_probability,
        risk_tier=tier,
        risk_score=RISK_SCORES[tier],
        primary_risk_factor=primary_factor,
        rate_adjustment_factor=adjustment,
        recommended_monthly_rate=recommended_rate,
        recommendation_reason=reason,
    )


def connect(database_url: str):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run: pip install -r requirements.txt")
    return psycopg2.connect(database_url)


def ensure_quote_table(cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS "QuoteRecommendation" (
          "QuoteRecommendationID" INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
          "CustomerID" INT NOT NULL REFERENCES "Customer" ("CustomerID"),
          "HealthProfileID" INT NOT NULL REFERENCES "CustomerHealthProfile" ("HealthProfileID"),
          "RiskTier" varchar(20) NOT NULL,
          "BaseMonthlyRate" decimal(12,2) NOT NULL,
          "RateAdjustmentFactor" decimal(5,2) NOT NULL,
          "RecommendedMonthlyRate" decimal(12,2) NOT NULL,
          "RecommendationReason" varchar(255),
          "CreatedDate" date NOT NULL
        );
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_quoterecommendation_customer
        ON "QuoteRecommendation" ("CustomerID");
        """
    )


def upsert_disease(cursor) -> int:
    cursor.execute(
        """
        INSERT INTO "ChronicDisease" ("DiseaseName", "DiseaseCategory", "Description")
        VALUES (%s, %s, %s)
        ON CONFLICT ("DiseaseName")
        DO UPDATE SET
          "DiseaseCategory" = EXCLUDED."DiseaseCategory",
          "Description" = EXCLUDED."Description"
        RETURNING "DiseaseID";
        """,
        (
            "Lifestyle Chronic Disease Risk",
            "Lifestyle Risk",
            "Risk tier used for insurance quote support and underwriting review.",
        ),
    )
    return int(cursor.fetchone()[0])


def upsert_risk_factor(cursor, factor_name: str) -> int:
    cursor.execute(
        """
        INSERT INTO "RiskFactor" ("FactorName", "FactorCategory", "IsChangeable", "Description")
        VALUES (%s, %s, %s, %s)
        ON CONFLICT ("FactorName")
        DO UPDATE SET
          "FactorCategory" = EXCLUDED."FactorCategory",
          "IsChangeable" = EXCLUDED."IsChangeable",
          "Description" = EXCLUDED."Description"
        RETURNING "RiskFactorID";
        """,
        (
            factor_name,
            "Lifestyle",
            factor_name not in ("Age", "Heredity"),
            "Primary lifestyle factor selected by the quote support workflow.",
        ),
    )
    return int(cursor.fetchone()[0])


def insert_workflow_result(
    database_url: str,
    customer: CustomerInput,
    result: RiskResult,
    refresh_view: bool,
) -> dict[str, int | None]:
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            ensure_quote_table(cursor)
            disease_id = upsert_disease(cursor)
            risk_factor_id = upsert_risk_factor(cursor, result.primary_risk_factor)

            cursor.execute(
                """
                INSERT INTO "Customer"
                  ("CustomerType", "FirstName", "LastName", "Status")
                VALUES (%s, %s, %s, %s)
                RETURNING "CustomerID";
                """,
                ("Individual", customer.first_name, customer.last_name, "Active"),
            )
            customer_id = int(cursor.fetchone()[0])

            notes = (
                f"Quote support tier={result.risk_tier}; "
                f"base=${decimal_money(customer.base_monthly_rate)}; "
                f"adjustment={result.rate_adjustment_factor}; "
                f"recommended=${result.recommended_monthly_rate}."
            )
            cursor.execute(
                """
                INSERT INTO "CustomerHealthProfile"
                  ("CustomerID", "DiseaseID", "ChronicDiseaseRiskScore",
                   "PrimaryRiskFactorID", "AssessmentDate", "AssessmentMethod", "Notes")
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING "HealthProfileID";
                """,
                (
                    customer_id,
                    disease_id,
                    result.risk_score,
                    risk_factor_id,
                    date.today(),
                    "Lifestyle risk decision tree model",
                    notes,
                ),
            )
            health_profile_id = int(cursor.fetchone()[0])

            cursor.execute(
                """
                INSERT INTO "HealthDataLakeRef"
                  ("HealthProfileID", "SourceSystem", "DataFormat", "CloudStorageURI",
                   "DatasetName", "LoadDate", "FileDescription")
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING "DataLakeRefID";
                """,
                (
                    health_profile_id,
                    "Class ML Pipeline",
                    "CSV",
                    DEFAULT_DATASET_URI,
                    "chronic_disease_sample_data.csv",
                    date.today(),
                    "Training data reference for lifestyle risk quote support.",
                ),
            )
            data_lake_ref_id = int(cursor.fetchone()[0])

            cursor.execute(
                """
                INSERT INTO "QuoteRecommendation"
                  ("CustomerID", "HealthProfileID", "RiskTier", "BaseMonthlyRate",
                   "RateAdjustmentFactor", "RecommendedMonthlyRate",
                   "RecommendationReason", "CreatedDate")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING "QuoteRecommendationID";
                """,
                (
                    customer_id,
                    health_profile_id,
                    result.risk_tier,
                    decimal_money(customer.base_monthly_rate),
                    result.rate_adjustment_factor,
                    result.recommended_monthly_rate,
                    result.recommendation_reason,
                    date.today(),
                ),
            )
            quote_recommendation_id = int(cursor.fetchone()[0])

            if refresh_view:
                try:
                    cursor.execute('REFRESH MATERIALIZED VIEW "CustomerHealthRiskSummary";')
                except Exception as exc:
                    print(f"Skipped materialized view refresh: {exc}")
                    conn.rollback()
                    return {
                        "customer_id": customer_id,
                        "health_profile_id": health_profile_id,
                        "data_lake_ref_id": data_lake_ref_id,
                        "quote_recommendation_id": quote_recommendation_id,
                        "summary_rows": None,
                    }

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM "CustomerHealthProfile"
                WHERE "CustomerID" = %s;
                """,
                (customer_id,),
            )
            summary_rows = int(cursor.fetchone()[0])

        conn.commit()

    return {
        "customer_id": customer_id,
        "health_profile_id": health_profile_id,
        "data_lake_ref_id": data_lake_ref_id,
        "quote_recommendation_id": quote_recommendation_id,
        "summary_rows": summary_rows,
    }


def print_result(customer: CustomerInput, result: RiskResult) -> None:
    print()
    print("Lifestyle Risk-Based Quote Support Result")
    print("-----------------------------------------")
    print(f"Customer: {customer.first_name} {customer.last_name}")
    print(f"Model high-risk prediction: {result.model_prediction}")
    print(f"High-risk probability: {result.risk_probability:.2f}")
    print(f"Risk tier: {result.risk_tier}")
    print(f"Primary risk factor: {result.primary_risk_factor}")
    print(f"Base monthly rate: ${decimal_money(customer.base_monthly_rate)}")
    print(f"Rate adjustment factor: {result.rate_adjustment_factor}")
    print(f"Recommended monthly rate: ${result.recommended_monthly_rate}")
    print(f"Recommendation reason: {result.recommendation_reason}")


def main() -> None:
    args = parse_args()
    customer = read_customer_input(args)
    validate_customer_input(customer)

    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run retrain_model.py first."
        )

    model = joblib.load(model_path)
    result = score_risk(model, customer)
    print_result(customer, result)

    if args.dry_run:
        print()
        print("Dry run complete. No database rows were written.")
        return

    if not args.database_url:
        raise RuntimeError(
            "No database URL found. Set DMS_DATABASE_URL or PG_URI, or run with --dry-run."
        )

    ids = insert_workflow_result(
        database_url=args.database_url,
        customer=customer,
        result=result,
        refresh_view=not args.skip_refresh_view,
    )
    print()
    print("Database update complete")
    print("------------------------")
    print(f"CustomerID: {ids['customer_id']}")
    print(f"HealthProfileID: {ids['health_profile_id']}")
    print(f"DataLakeRefID: {ids['data_lake_ref_id']}")
    print(f"QuoteRecommendationID: {ids['quote_recommendation_id']}")
    print(f"CustomerHealthProfile rows for customer: {ids['summary_rows']}")


if __name__ == "__main__":
    main()
