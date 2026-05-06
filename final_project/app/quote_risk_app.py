from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import joblib
import pandas as pd

from db_orm import (
    Customer,
    CustomerHealthProfile,
    HealthDataLakeRef,
    QuoteRecommendation,
    count_customer_health_profiles,
    make_session,
    refresh_health_summary,
    upsert_disease,
    upsert_risk_factor,
)
from parse_unstructured_notes import extract_features


PART4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PART4_ROOT / "model" / "chronic_disease_risk_model.joblib"
DEFAULT_RATE_PATH = PART4_ROOT / "model" / "cms_rate_benchmark.csv"
DEFAULT_NOTES_PATH = PART4_ROOT / "data_unstructured" / "patient_lifestyle_notes.jsonl"
DEFAULT_DATASET_URI = (
    "Centers for Disease Control and Prevention (CDC) BRFSS 2023: "
    "https://www.cdc.gov/brfss/annual_data/annual_2023.html; "
    "Centers for Medicare & Medicaid Services (CMS) Rate PUF 2024: "
    "https://www.cms.gov/marketplace/resources/data/public-use-files; "
    "Unstructured lifestyle notes: local://final_project/data_unstructured/"
    "patient_lifestyle_notes.jsonl"
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
    raw_lifestyle_note: str | None = None


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


def prompt_str_if_missing(value: str | None, prompt: str) -> str:
    if value is not None:
        return value
    return input(prompt).strip()


def prompt_int_if_missing(value: int | None, prompt: str) -> int:
    if value is not None:
        return value
    return int(input(prompt).strip())


def prompt_yes_no_if_missing(value: int | None, prompt: str) -> int:
    if value is not None:
        return value

    while True:
        raw = input(f"{prompt} (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return 1
        if raw in ("n", "no"):
            return 0
        print("Please enter y or n.")


def prompt_general_health_if_missing(value: int | None) -> int:
    if value is not None:
        return value

    print("General health:")
    print("  1 = excellent")
    print("  2 = very good")
    print("  3 = good")
    print("  4 = fair")
    print("  5 = poor")
    return int(input("Choose 1-5: ").strip())


def load_note_defaults(notes_path: Path, note_id: str | None) -> dict[str, object]:
    if not note_id:
        return {}

    with notes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["patient_note_id"]).lower() == note_id.lower():
                features = extract_features(str(row["note"]))
                return {
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "age": int(row["age"]),
                    **features,
                    "raw_lifestyle_note": row["note"],
                }

    raise ValueError(f"Patient note ID '{note_id}' was not found in {notes_path}.")


def default_str(defaults: dict[str, object], key: str) -> str | None:
    value = defaults.get(key)
    return value if isinstance(value, str) else None


def default_int(defaults: dict[str, object], key: str) -> int | None:
    value = defaults.get(key)
    return value if isinstance(value, int) else None


def optional_note_id_if_missing(value: str | None) -> str | None:
    if value:
        return value
    raw = input("Patient note ID from unstructured notes file (optional, press Enter to skip): ")
    return raw.strip() or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BRFSS and CMS based lifestyle risk rate recommendation demo."
    )
    parser.add_argument("--first-name")
    parser.add_argument("--last-name")
    parser.add_argument(
        "--note-id",
        help="Optional patient_note_id from data_unstructured/patient_lifestyle_notes.jsonl.",
    )
    parser.add_argument("--notes-path", type=Path, default=DEFAULT_NOTES_PATH)
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
    note_id = optional_note_id_if_missing(args.note_id)
    note_defaults = load_note_defaults(args.notes_path.resolve(), note_id)

    return CustomerInput(
        first_name=prompt_str_if_missing(
            args.first_name or default_str(note_defaults, "first_name"),
            "First name: ",
        ),
        last_name=prompt_str_if_missing(
            args.last_name or default_str(note_defaults, "last_name"),
            "Last name: ",
        ),
        age=prompt_int_if_missing(args.age or default_int(note_defaults, "age"), "Age: "),
        tobacco_user=prompt_yes_no_if_missing(
            args.tobacco_user
            if args.tobacco_user is not None
            else default_int(note_defaults, "tobacco_user"),
            "Tobacco user?",
        ),
        obese=prompt_yes_no_if_missing(
            args.obese if args.obese is not None else default_int(note_defaults, "obese"),
            "Obese BMI category?",
        ),
        physical_inactivity=prompt_yes_no_if_missing(
            args.physical_inactivity
            if args.physical_inactivity is not None
            else default_int(note_defaults, "physical_inactivity"),
            "Physically inactive?",
        ),
        binge_drinking=prompt_yes_no_if_missing(
            args.binge_drinking
            if args.binge_drinking is not None
            else default_int(note_defaults, "binge_drinking"),
            "Binge drinking risk?",
        ),
        heavy_drinking=prompt_yes_no_if_missing(
            args.heavy_drinking
            if args.heavy_drinking is not None
            else default_int(note_defaults, "heavy_drinking"),
            "Heavy drinking risk?",
        ),
        diabetes=prompt_yes_no_if_missing(
            args.diabetes
            if args.diabetes is not None
            else default_int(note_defaults, "diabetes"),
            "Diabetes?",
        ),
        general_health=prompt_general_health_if_missing(
            args.general_health
            if args.general_health is not None
            else default_int(note_defaults, "general_health")
        ),
        manual_base_monthly_rate=args.base_monthly_rate,
        raw_lifestyle_note=(
            str(note_defaults["raw_lifestyle_note"])
            if "raw_lifestyle_note" in note_defaults
            else None
        ),
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


def insert_workflow_result(
    database_url: str,
    customer: CustomerInput,
    result: RiskResult,
    refresh_view: bool,
) -> dict[str, int | None]:
    with make_session(database_url) as session:
        disease_id = upsert_disease(session)
        risk_factor_id = upsert_risk_factor(session, result.primary_risk_factor)

        db_customer = Customer(
            CustomerType="Individual",
            FirstName=customer.first_name,
            LastName=customer.last_name,
            Status="Active",
        )
        session.add(db_customer)
        session.flush()
        customer_id = int(db_customer.CustomerID)

        notes = (
            f"Rate recommendation tier={result.risk_tier}; "
            f"base=${result.base_monthly_rate}; "
            f"adjustment={result.rate_adjustment_factor}; "
            f"recommended=${result.recommended_monthly_rate}; "
            f"source={result.base_rate_source}."
        )
        if customer.raw_lifestyle_note:
            notes = f"{notes} Unstructured lifestyle note parsed."
        db_health_profile = CustomerHealthProfile(
            CustomerID=customer_id,
            DiseaseID=disease_id,
            ChronicDiseaseRiskScore=result.risk_score,
            PrimaryRiskFactorID=risk_factor_id,
            AssessmentDate=date.today(),
            AssessmentMethod="CDC BRFSS decision tree lifestyle risk model",
            Notes=notes,
        )
        session.add(db_health_profile)
        session.flush()
        health_profile_id = int(db_health_profile.HealthProfileID)

        db_data_lake_ref = HealthDataLakeRef(
            HealthProfileID=health_profile_id,
            SourceSystem="CDC BRFSS / CMS Rate PUF / Unstructured Notes Workflow",
            DataFormat="XPT/CSV",
            CloudStorageURI=DEFAULT_DATASET_URI,
            DatasetName=(
                "brfss_lifestyle_risk_training.csv; cms_rate_benchmark.csv; "
                "patient_lifestyle_notes.jsonl"
            ),
            LoadDate=date.today(),
            FileDescription=(
                "CDC BRFSS lifestyle training data, CMS Exchange Rate PUF benchmark, "
                "and parsed unstructured lifestyle note input."
            ),
        )
        session.add(db_data_lake_ref)
        session.flush()
        data_lake_ref_id = int(db_data_lake_ref.DataLakeRefID)

        db_quote_recommendation = QuoteRecommendation(
            CustomerID=customer_id,
            HealthProfileID=health_profile_id,
            RiskTier=result.risk_tier,
            BaseMonthlyRate=result.base_monthly_rate,
            RateAdjustmentFactor=result.rate_adjustment_factor,
            RecommendedMonthlyRate=result.recommended_monthly_rate,
            RecommendationReason=result.recommendation_reason,
            CreatedDate=date.today(),
        )
        session.add(db_quote_recommendation)
        session.flush()
        quote_recommendation_id = int(db_quote_recommendation.QuoteRecommendationID)

        summary_rows = count_customer_health_profiles(session, customer_id)
        if refresh_view:
            try:
                refresh_health_summary(session)
            except Exception as exc:
                print(f"Skipped materialized view refresh: {exc}")
                session.rollback()
                return {
                    "customer_id": customer_id,
                    "health_profile_id": health_profile_id,
                    "data_lake_ref_id": data_lake_ref_id,
                    "quote_recommendation_id": quote_recommendation_id,
                    "summary_rows": None,
                }

        session.commit()

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
    if customer.raw_lifestyle_note:
        print(f"Unstructured lifestyle note: {customer.raw_lifestyle_note}")
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
