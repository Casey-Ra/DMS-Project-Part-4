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


PART4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PART4_ROOT / "model" / "chronic_disease_risk_model.joblib"
DEFAULT_RATE_PATH = PART4_ROOT / "model" / "cms_rate_benchmark.csv"
DEFAULT_DATASET_URI = (
    "Centers for Disease Control and Prevention (CDC) BRFSS 2023: "
    "https://www.cdc.gov/brfss/annual_data/annual_2023.html; "
    "Centers for Medicare & Medicaid Services (CMS) Rate PUF 2024: "
    "https://www.cms.gov/marketplace/resources/data/public-use-files"
)

FEATURE_COLUMNS = [
    "age",
    "tobacco_user",
    "obese",
    "physical_inactivity",
    "binge_drinking",
    "heavy_drinking",
    "diabetes",
    "general_health",
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
    tobacco_user: int
    obese: int
    physical_inactivity: int
    binge_drinking: int
    heavy_drinking: int
    diabetes: int
    general_health: int
    manual_base_monthly_rate: Decimal | None


@dataclass
class RateBenchmark:
    source: str
    base_monthly_rate: Decimal


@dataclass
class RiskResult:
    model_prediction: int
    risk_probability: float
    risk_tier: str
    risk_score: Decimal
    primary_risk_factor: str
    base_rate_source: str
    base_monthly_rate: Decimal
    rate_adjustment_factor: Decimal
    recommended_monthly_rate: Decimal
    recommendation_reason: str


def decimal_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def prompt_if_missing(value: object | None, prompt: str, cast=str):
    if value is not None:
        return value
    raw = input(prompt).strip()
    if raw == "" and cast is Decimal:
        return None
    return cast(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BRFSS and CMS based lifestyle risk rate recommendation demo."
    )
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument("--age", type=int)
    parser.add_argument("--tobacco-user", type=int, choices=(0, 1))
    parser.add_argument("--obese", type=int, choices=(0, 1))
    parser.add_argument("--physical-inactivity", type=int, choices=(0, 1))
    parser.add_argument("--binge-drinking", type=int, choices=(0, 1))
    parser.add_argument("--heavy-drinking", type=int, choices=(0, 1))
    parser.add_argument("--diabetes", type=int, choices=(0, 1))
    parser.add_argument(
        "--general-health",
        type=int,
        choices=range(1, 6),
        help="BRFSS general health scale: 1=excellent, 5=poor.",
    )
    parser.add_argument(
        "--base-monthly-rate",
        type=Decimal,
        help="Optional manual base rate. If omitted, CMS Rate PUF median rate is used.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rate-path", type=Path, default=DEFAULT_RATE_PATH)
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
        tobacco_user=prompt_if_missing(args.tobacco_user, "Tobacco user? (0=no, 1=yes): ", int),
        obese=prompt_if_missing(args.obese, "Obese BMI category? (0=no, 1=yes): ", int),
        physical_inactivity=prompt_if_missing(
            args.physical_inactivity, "Physically inactive? (0=no, 1=yes): ", int
        ),
        binge_drinking=prompt_if_missing(
            args.binge_drinking, "Binge drinking risk? (0=no, 1=yes): ", int
        ),
        heavy_drinking=prompt_if_missing(
            args.heavy_drinking, "Heavy drinking risk? (0=no, 1=yes): ", int
        ),
        diabetes=prompt_if_missing(args.diabetes, "Diabetes? (0=no, 1=yes): ", int),
        general_health=prompt_if_missing(
            args.general_health, "General health (1=excellent, 5=poor): ", int
        ),
        manual_base_monthly_rate=args.base_monthly_rate,
    )


def validate_customer_input(customer: CustomerInput) -> None:
    if customer.age < 18 or customer.age > 80:
        raise ValueError("Age must be between 18 and 80 for the BRFSS/CMS demo.")
    for label, value in [
        ("tobacco user", customer.tobacco_user),
        ("obese", customer.obese),
        ("physical inactivity", customer.physical_inactivity),
        ("binge drinking", customer.binge_drinking),
        ("heavy drinking", customer.heavy_drinking),
        ("diabetes", customer.diabetes),
    ]:
        if value not in (0, 1):
            raise ValueError(f"{label.title()} must be 0 or 1.")
    if customer.general_health < 1 or customer.general_health > 5:
        raise ValueError("General health must be from 1 to 5.")
    if customer.manual_base_monthly_rate is not None and customer.manual_base_monthly_rate <= 0:
        raise ValueError("Base monthly rate must be greater than zero.")


def feature_frame(customer: CustomerInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": customer.age,
                "tobacco_user": customer.tobacco_user,
                "obese": customer.obese,
                "physical_inactivity": customer.physical_inactivity,
                "binge_drinking": customer.binge_drinking,
                "heavy_drinking": customer.heavy_drinking,
                "diabetes": customer.diabetes,
                "general_health": customer.general_health,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def load_rate_benchmark(rate_path: Path, customer: CustomerInput) -> RateBenchmark:
    if customer.manual_base_monthly_rate is not None:
        return RateBenchmark(
            source="Manual user-entered base rate",
            base_monthly_rate=decimal_money(customer.manual_base_monthly_rate),
        )

    if not rate_path.exists():
        raise FileNotFoundError(
            f"CMS rate benchmark not found: {rate_path}. "
            "Run build_real_data_sources.py first, or provide --base-monthly-rate."
        )

    rates = pd.read_csv(rate_path)
    nearest = rates.iloc[(rates["age"] - customer.age).abs().argsort()[:1]].iloc[0]
    column = "cms_tobacco_monthly_rate" if customer.tobacco_user else "cms_base_monthly_rate"
    return RateBenchmark(
        source=(
            "Centers for Medicare & Medicaid Services (CMS) 2024 Rate PUF "
            f"median {column} for age {int(nearest['age'])}"
        ),
        base_monthly_rate=decimal_money(Decimal(str(nearest[column]))),
    )


def identify_primary_risk_factor(customer: CustomerInput) -> str:
    if customer.diabetes:
        return "Diabetes"
    if customer.tobacco_user:
        return "Tobacco Use"
    if customer.obese:
        return "Obesity"
    if customer.physical_inactivity:
        return "Physical Inactivity"
    if customer.heavy_drinking:
        return "Heavy Drinking"
    if customer.binge_drinking:
        return "Binge Drinking"
    if customer.general_health >= 4:
        return "Fair/Poor General Health"
    if customer.age >= 55:
        return "Age"
    return "General Lifestyle"


def lifestyle_flag_count(customer: CustomerInput) -> int:
    return sum(
        [
            customer.age >= 55,
            customer.tobacco_user == 1,
            customer.obese == 1,
            customer.physical_inactivity == 1,
            customer.binge_drinking == 1,
            customer.heavy_drinking == 1,
            customer.diabetes == 1,
            customer.general_health >= 4,
        ]
    )


def score_risk(model, customer: CustomerInput, benchmark: RateBenchmark) -> RiskResult:
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
    recommended_rate = decimal_money(benchmark.base_monthly_rate * adjustment)
    reason = (
        f"{tier} rate recommendation based on the "
        "Centers for Disease Control and Prevention (CDC) BRFSS lifestyle risk model, "
        f"{risk_probability:.2f} high-risk probability, "
        "Centers for Medicare & Medicaid Services (CMS) base rate benchmark, "
        f"and {flags} customer risk flags."
    )

    return RiskResult(
        model_prediction=prediction,
        risk_probability=risk_probability,
        risk_tier=tier,
        risk_score=RISK_SCORES[tier],
        primary_risk_factor=identify_primary_risk_factor(customer),
        base_rate_source=benchmark.source,
        base_monthly_rate=benchmark.base_monthly_rate,
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
            "Risk tier used for insurance rate recommendation and underwriting review.",
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
            factor_name not in ("Age",),
            "Primary lifestyle factor selected by the rate recommendation workflow.",
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
                f"Rate recommendation tier={result.risk_tier}; "
                f"base=${result.base_monthly_rate}; "
                f"adjustment={result.rate_adjustment_factor}; "
                f"recommended=${result.recommended_monthly_rate}; "
                f"source={result.base_rate_source}."
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
                    "CDC BRFSS decision tree lifestyle risk model",
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
                    "CDC BRFSS / CMS Rate PUF Workflow",
                    "XPT/CSV",
                    DEFAULT_DATASET_URI,
                    "brfss_lifestyle_risk_training.csv; cms_rate_benchmark.csv",
                    date.today(),
                    "CDC BRFSS lifestyle training data and CMS Exchange Rate PUF benchmark.",
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
                    result.base_monthly_rate,
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
    print("Lifestyle Risk-Based Rate Recommendation Result")
    print("-----------------------------------------------")
    print(f"Customer: {customer.first_name} {customer.last_name}")
    print(f"Age: {customer.age}")
    print(f"Tobacco user: {customer.tobacco_user}")
    print(f"Model high-risk prediction: {result.model_prediction}")
    print(f"High-risk probability: {result.risk_probability:.2f}")
    print(f"Risk tier: {result.risk_tier}")
    print(f"Primary risk factor: {result.primary_risk_factor}")
    print(f"Base monthly rate: ${result.base_monthly_rate}")
    print(f"Base rate source: {result.base_rate_source}")
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
    benchmark = load_rate_benchmark(args.rate_path.resolve(), customer)
    result = score_risk(model, customer, benchmark)
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
