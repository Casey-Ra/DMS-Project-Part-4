from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PART4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTES_PATH = PART4_ROOT / "data_unstructured" / "patient_lifestyle_notes.jsonl"
DEFAULT_OUTPUT_PATH = PART4_ROOT / "model" / "unstructured_note_features.csv"


def text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def text_has_negated(text: str, terms: tuple[str, ...]) -> bool:
    negations = ("no ", "not ", "denies ", "non-", "without ")
    return any(f"{negation}{term}" in text for negation in negations for term in terms)


def extract_binary(text: str, positive_terms: tuple[str, ...], negative_terms: tuple[str, ...]) -> int:
    normalized = text.lower()
    if text_has_any(normalized, negative_terms) or text_has_negated(normalized, positive_terms):
        return 0
    if text_has_any(normalized, positive_terms):
        return 1
    return 0


def extract_general_health(text: str) -> int:
    normalized = text.lower()
    if "poor" in normalized:
        return 5
    if "fair" in normalized:
        return 4
    if "good" in normalized and "very good" not in normalized:
        return 3
    if "very good" in normalized:
        return 2
    if "excellent" in normalized:
        return 1
    return 3


def extract_features(note: str) -> dict[str, int]:
    return {
        "tobacco_user": extract_binary(
            note,
            ("smokes", "smoking", "smoker", "tobacco", "cigarette", "cigarettes"),
            ("no smoking", "no tobacco", "non-smoker", "does not smoke", "denies smoking"),
        ),
        "obese": extract_binary(
            note,
            ("obesity", "obese", "high bmi"),
            ("no obesity", "not obese", "healthy bmi", "healthy weight"),
        ),
        "physical_inactivity": extract_binary(
            note,
            ("inactive", "sedentary", "low physical activity", "limited exercise", "low activity"),
            ("active", "regular exercise", "exercises regularly", "exercises frequently"),
        ),
        "binge_drinking": extract_binary(
            note,
            ("binge drinking", "binge"),
            ("no binge drinking", "no alcohol risk", "denies alcohol"),
        ),
        "heavy_drinking": extract_binary(
            note,
            ("heavy drinking", "heavy alcohol", "alcohol misuse"),
            (
                "no heavy drinking",
                "no heavy alcohol",
                "no alcohol misuse",
                "denies alcohol misuse",
                "denies tobacco use and alcohol misuse",
            ),
        ),
        "diabetes": extract_binary(
            note,
            ("diabetes", "diabetic"),
            ("no diabetes", "denies diabetes", "not diabetes", "prediabetes"),
        ),
        "general_health": extract_general_health(note),
    }


def load_notes(notes_path: Path) -> list[dict[str, object]]:
    rows = []
    with notes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract model features from unstructured lifestyle notes."
    )
    parser.add_argument("--notes-path", type=Path, default=DEFAULT_NOTES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    extracted_rows = []
    for row in load_notes(args.notes_path):
        features = extract_features(str(row["note"]))
        extracted_rows.append(
            {
                "patient_note_id": row["patient_note_id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "age": row["age"],
                **features,
                "raw_note": row["note"],
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(extracted_rows).to_csv(args.output_path, index=False)

    print("Unstructured lifestyle notes parsed")
    print("-----------------------------------")
    print(f"Notes parsed: {len(extracted_rows)}")
    print(f"Source: {args.notes_path.resolve()}")
    print(f"Saved: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
