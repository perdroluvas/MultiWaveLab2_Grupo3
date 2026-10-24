"""
Build a new feature representation from the original MultiWaveLab2 features.

The script follows the same conventions of the rest of the project: the
experiment is described by a JSON configuration file and the new
representation is applied to the features CSV produced by the pipeline.

Usage
-----
From the ``src`` folder of the repository::

    python ../new_features/run_feature_representation.py \
        ../config_experiments/input_ieb1/exp0.json

Options::

    --spec SPEC.json      specification of the representations (optional)
    --repr a,b,c          shortcut to select the representations
    --source SOURCE       fused | train_test
    --suffix SUFFIX       suffix of the generated file (default: _repr)
    --no-original         do not keep the original columns in the output
    --describe            only describe the original feature matrix

The specification can also be written directly inside the experiment JSON,
under the ``FEATURE_REPRESENTATION`` key::

    {
      "FEATURE_REPRESENTATION": {
        "representations": ["ratio"],
        "keep_original": true
      }
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import ensure_import_paths

ensure_import_paths()

from datasets_util.naming_conventions import DatasetConfig
from feature_representation import (
    DEFAULT_SPEC,
    KEEP_ORIGINAL_KEY,
    REPRESENTATIONS_KEY,
    TARGET_COLUMN,
    apply_representation,
    describe_feature_matrix,
)


def load_spec(
    dataset_config: DatasetConfig,
    spec_file: str | None,
    representations: str | None,
    keep_original: bool,
) -> dict[str, Any]:
    """
    Build the specification of the representations.

    Priority: command line options, then the ``FEATURE_REPRESENTATION`` key of
    the experiment JSON, then the default specification.
    """
    spec: dict[str, Any] = {}

    try:
        from_experiment = dataset_config.get_value("FEATURE_REPRESENTATION", {})
    except KeyError:
        from_experiment = {}
    if isinstance(from_experiment, dict):
        spec.update(from_experiment)

    if spec_file:
        with open(spec_file, "r", encoding="utf-8") as handle:
            from_file = json.load(handle)
        if not isinstance(from_file, dict):
            raise ValueError("The specification file must contain a JSON object.")
        # The specification file may follow the same convention of the
        # experiment JSON, where everything is inside a
        # "FEATURE_REPRESENTATION" key.
        if set(from_file.keys()) == {"FEATURE_REPRESENTATION"}:
            from_file = from_file["FEATURE_REPRESENTATION"]
        spec.update(from_file)

    if representations:
        spec[REPRESENTATIONS_KEY] = [
            name.strip() for name in representations.split(",") if name.strip()
        ]

    if not keep_original:
        spec[KEEP_ORIGINAL_KEY] = False

    return spec


def print_report(report: dict[str, Any]) -> None:
    """Print a human readable summary of what was done."""
    print()
    print("=" * 78)
    print("FEATURE REPRESENTATION REPORT")
    print("=" * 78)
    print(f"original features : {report['n_input_features']}")
    print(f"original rows     : {report['n_input_rows']}")
    print(f"output rows       : {report.get('n_output_rows', 'n/a')}")
    print(f"output features   : {report.get('n_output_features', 'n/a')}")
    print(f"new columns       : {len(report.get('new_columns', []))}")

    print("\napplied representations:")
    for entry in report.get("applied", []):
        print(f"  - {entry['name']:16s} {entry['n_columns']:5d} columns")

    if report.get("skipped"):
        print("\nskipped rules:")
        for entry in report["skipped"]:
            missing = entry.get("missing") or []
            detail = f" (missing: {', '.join(missing)})" if missing else ""
            print(
                f"  - {entry['name']}/{entry.get('rule', '?')}: "
                f"{entry.get('reason', '')}{detail}"
            )


def run_fused(
    dataset_config: DatasetConfig,
    spec: dict[str, Any],
    suffix: str,
    sampling_rate: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the representation to the fused features file."""
    features_folder = Path(dataset_config.get_dataset_features_path())
    input_file = features_folder / dataset_config.get_features_file_name()
    if not input_file.exists():
        raise FileNotFoundError(
            f"Features file not found: {input_file}\n"
            "Run the feature extraction stage of the pipeline first "
            "(see section 4.1 of new_features/README.md)."
        )
    print(f"Reading original features from: {input_file}")
    df = pd.read_csv(input_file)
    result, report = apply_representation(df, spec, sampling_rate=sampling_rate)
    output_file = features_folder / (
        f"{Path(dataset_config.get_features_file_name()).stem}{suffix}.csv"
    )
    result.to_csv(output_file, index=False)
    print(f"New representation saved to: {output_file}")
    report["output_file"] = str(output_file)
    return result, report


def run_train_test(
    dataset_config: DatasetConfig,
    spec: dict[str, Any],
    suffix: str,
    sampling_rate: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply the representation to the train and test splits.

    The representation is computed independently for the training and the test
    split, so no information of the test set leaks into the model.
    """
    features_folder = Path(dataset_config.get_dataset_features_path())
    train_file = Path(dataset_config.get_splitted_features_file_name("train"))
    test_file = Path(dataset_config.get_splitted_features_file_name("test"))

    for path in (train_file, test_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Split features file not found: {path}\n"
                "Run the train/test split stage of the pipeline first "
                "(see section 4.1 of new_features/README.md)."
            )

    print(f"Reading training features from: {train_file}")
    df_train = pd.read_csv(train_file)
    print(f"Reading test features from: {test_file}")
    df_test = pd.read_csv(test_file)

    result_train, report = apply_representation(
        df_train, spec, sampling_rate=sampling_rate
    )
    result_test, _ = apply_representation(
        df_test, spec, sampling_rate=sampling_rate
    )

    train_out = features_folder / f"{train_file.stem}{suffix}.csv"
    test_out = features_folder / f"{test_file.stem}{suffix}.csv"
    result_train.to_csv(train_out, index=False)
    result_test.to_csv(test_out, index=False)
    print(f"New representation saved to: {train_out}")
    print(f"New representation saved to: {test_out}")
    report["output_file"] = str(train_out)
    report["output_file_test"] = str(test_out)
    report["n_test_rows"] = int(result_test.shape[0])
    return result_train, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a new feature representation from the original features."
    )
    parser.add_argument(
        "json_file",
        metavar="JSON_FILE",
        help="Path to the experiment JSON configuration file.",
    )
    parser.add_argument(
        "--spec",
        metavar="SPEC_FILE",
        default=None,
        help="Path to a JSON file describing the representations.",
    )
    parser.add_argument(
        "--repr",
        metavar="REPRESENTATIONS",
        default=None,
        help=(
            "Comma separated list of representations. "
            "The presentation uses only: ratio"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("fused", "train_test"),
        default="fused",
        help="Apply the representation to the fused file or to the train/test splits.",
    )
    parser.add_argument(
        "--suffix",
        default="_repr",
        help="Suffix of the generated CSV file.",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="Do not keep the original columns in the output file.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Only describe the original feature matrix and exit.",
    )
    args = parser.parse_args()

    dataset_config = DatasetConfig(args.json_file)
    spec = load_spec(
        dataset_config, args.spec, args.repr, not args.no_original
    )
    if not spec.get(REPRESENTATIONS_KEY):
        spec[REPRESENTATIONS_KEY] = list(DEFAULT_SPEC[REPRESENTATIONS_KEY])

    print("Representations requested:", spec[REPRESENTATIONS_KEY])

    sampling_rate: float | None = None
    try:
        sampling_rate = float(dataset_config.get_ppg_fs())
    except (TypeError, ValueError):
        sampling_rate = None

    if args.describe:
        features_folder = Path(dataset_config.get_dataset_features_path())
        input_file = features_folder / dataset_config.get_features_file_name()
        if not input_file.exists():
            raise FileNotFoundError(f"Features file not found: {input_file}")
        summary = describe_feature_matrix(pd.read_csv(input_file))
        print(summary.to_string(index=False))
        return

    if args.source == "train_test":
        result, report = run_train_test(
            dataset_config, spec, args.suffix, sampling_rate
        )
    else:
        result, report = run_fused(
            dataset_config, spec, args.suffix, sampling_rate
        )

    print_report(report)

    stats_folder = Path(dataset_config.get_statistics_output_folder())
    stats_folder.mkdir(parents=True, exist_ok=True)
    report_file = stats_folder / "feature_representation_report.json"
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")

    try:
        numeric = result.select_dtypes(include=[np.number])
        if TARGET_COLUMN in numeric.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                correlations = (
                    numeric.corrwith(numeric[TARGET_COLUMN])
                    .drop(TARGET_COLUMN)
                    .abs()
                )
            print(f"\ntop 15 columns by |correlation| with {TARGET_COLUMN}:")
            print(correlations.sort_values(ascending=False).head(15).to_string())
    except (TypeError, ValueError) as error:
        print(f"\nCould not compute the correlation summary: {error}")


if __name__ == "__main__":
    main()
