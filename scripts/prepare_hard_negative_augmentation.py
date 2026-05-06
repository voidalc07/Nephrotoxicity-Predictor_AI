from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem


@dataclass(frozen=True)
class CandidateStatus:
    drug_name: str
    canonical_smiles: str
    label: int
    primary_toxicity: str
    source: str
    status: str
    existing_label: str


def canonicalize_smiles(smiles: str) -> str | None:
    # -------------------------------------------------------------------------
    # Canonical SMILES Normalisation
    # Hard-negative augmentation only makes sense if duplicate and conflicting
    # molecules are identified on a canonicalised structural representation
    # rather than by raw text string equality.
    # -------------------------------------------------------------------------
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def load_training_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    # Load the current benchmark training set and build a canonical-SMILES
    # lookup so new candidate negatives can be screened for collisions.
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    canonical_to_label: dict[str, int] = {}
    for row in rows:
        canonical = canonicalize_smiles(row["canonical SMILES"])
        if canonical is None:
            continue
        canonical_to_label[canonical] = int(row["label"])
    return rows, canonical_to_label


def load_candidate_statuses(path: Path, training_labels: dict[str, int]) -> list[CandidateStatus]:
    # -------------------------------------------------------------------------
    # Hard-Negative Triage
    # Candidates are classified as new, duplicate, conflicting, or invalid so
    # augmentation remains explicit and reversible. This supports the scientific
    # motivation of adding complex non-nephrotoxic molecules without silently
    # corrupting the original benchmark labels.
    # -------------------------------------------------------------------------
    statuses: list[CandidateStatus] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canonical = canonicalize_smiles(row["canonical_smiles"])
            if canonical is None:
                statuses.append(
                    CandidateStatus(
                        drug_name=row["drug_name"],
                        canonical_smiles=row["canonical_smiles"],
                        label=int(row["label"]),
                        primary_toxicity=row["primary_toxicity"],
                        source=row["source"],
                        status="invalid_smiles",
                        existing_label="",
                    )
                )
                continue

            existing_label = training_labels.get(canonical)
            if existing_label is None:
                status = "new"
            elif existing_label == int(row["label"]):
                status = "duplicate_same_label"
            else:
                status = "conflict_existing_label"

            statuses.append(
                CandidateStatus(
                    drug_name=row["drug_name"],
                    canonical_smiles=canonical,
                    label=int(row["label"]),
                    primary_toxicity=row["primary_toxicity"],
                    source=row["source"],
                    status=status,
                    existing_label="" if existing_label is None else str(existing_label),
                )
            )
    return statuses


def write_filtered_candidates(path: Path, statuses: list[CandidateStatus]) -> None:
    # Persist only the genuinely new hard negatives for later inspection or
    # manual curation.
    rows = [status for status in statuses if status.status == "new"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["drug_name", "canonical_smiles", "label", "primary_toxicity", "source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "drug_name": row.drug_name,
                    "canonical_smiles": row.canonical_smiles,
                    "label": row.label,
                    "primary_toxicity": row.primary_toxicity,
                    "source": row.source,
                }
            )


def write_status_report(path: Path, statuses: list[CandidateStatus]) -> None:
    # The status report is an audit trail showing which candidate molecules were
    # accepted, rejected as duplicates, or flagged as label conflicts.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "drug_name",
                "canonical_smiles",
                "label",
                "primary_toxicity",
                "source",
                "status",
                "existing_label",
            ],
        )
        writer.writeheader()
        for row in statuses:
            writer.writerow(
                {
                    "drug_name": row.drug_name,
                    "canonical_smiles": row.canonical_smiles,
                    "label": row.label,
                    "primary_toxicity": row.primary_toxicity,
                    "source": row.source,
                    "status": row.status,
                    "existing_label": row.existing_label,
                }
            )


def write_augmented_training_csv(
    path: Path,
    training_rows: list[dict[str, str]],
    statuses: list[CandidateStatus],
) -> None:
    # Build a deterministic augmented training CSV by appending only the novel
    # negative examples after the original benchmark rows.
    additions = [status for status in statuses if status.status == "new"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical SMILES", "label"])
        writer.writeheader()
        for row in training_rows:
            writer.writerow(
                {
                    "canonical SMILES": row["canonical SMILES"],
                    "label": row["label"],
                }
            )
        for row in additions:
            writer.writerow(
                {
                    "canonical SMILES": row.canonical_smiles,
                    "label": row.label,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    # CLI surface for reproducible augmentation file generation.
    parser = argparse.ArgumentParser(description="Prepare reproducible hard-negative augmentation files.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--filtered-csv", required=True)
    parser.add_argument("--status-report", required=True)
    parser.add_argument("--augmented-train-csv", required=True)
    return parser


def main() -> int:
    # -------------------------------------------------------------------------
    # Minimal Human-Curated Hard-Negative Workflow
    # This script operationalises the dissertation idea that false positives on
    # complex novel chemistry are best addressed by changing the training signal
    # rather than only tuning algorithms.
    # -------------------------------------------------------------------------
    parser = build_parser()
    args = parser.parse_args()

    training_rows, training_labels = load_training_rows(Path(args.train_csv))
    statuses = load_candidate_statuses(Path(args.candidate_csv), training_labels)

    write_filtered_candidates(Path(args.filtered_csv), statuses)
    write_status_report(Path(args.status_report), statuses)
    write_augmented_training_csv(Path(args.augmented_train_csv), training_rows, statuses)

    new_count = sum(1 for status in statuses if status.status == "new")
    duplicate_count = sum(1 for status in statuses if status.status == "duplicate_same_label")
    conflict_count = sum(1 for status in statuses if status.status == "conflict_existing_label")
    invalid_count = sum(1 for status in statuses if status.status == "invalid_smiles")
    print(
        f"Prepared hard-negative augmentation: {new_count} new, "
        f"{duplicate_count} duplicates, {conflict_count} conflicts, {invalid_count} invalid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
