"""
Construction of new feature representations from the original MultiWaveLab2 features.

The MultiWaveLab2 pipeline already extracts the "original" features from the
physiological signals (PPG, ECG and bioimpedance). See:

    src/signal_processing/ppg.py          -> groups p1, b2, s3, h4, u5, i6
    src/signal_processing/ecg.py          -> groups e1, e2, ...
    src/signal_processing/bioimpedance.py -> groups b1, b2, ...
    src/features/extract_features_for_all_modalities.py  (orchestration)

This module implements the next level of the feature hierarchy:

    level 1  sample     raw waveform
    level 2  pulse/beat features of a single pulse
    level 3  window     one feature vector per segment   <-- "original features"
    level 4  measurement one feature vector per glucose measurement
    level 5  participant one feature vector per participant

"Constructing a new feature representation from the original features" means
taking the level-3 matrix produced by the pipeline and re-expressing it as a
different object that is still usable by the machine learning stage.

Representations exposed
-----------------------
ratio            four dimensionless ratios between original features:
                 real heart rate, relative systolic width, normalized crest
                 time and the coefficient of variation of the pulse interval

Every representation is a pure function of the original feature matrix, so the
module can be applied to any dataset of the project and each transformation can
be evaluated independently.

Conventions
-----------
* The column naming convention of the project is preserved:
  ``<modality>_<group>_<feature>`` (see datasets_util/util_features.py).
* The derived columns created here use the prefix ``repr_`` so that they can
  never be confused with the original features.
* Metadata columns (participant_id, session_id, datetime, GLC, ...) are never
  used as predictors, except when a representation explicitly requires them
  (for example ``duration`` in the beat-rate consistency check).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12

TARGET_COLUMN = "GLC"

#: Key used to pass the sampling rate into the ratio expressions.
SAMPLING_RATE_KEY = "fs"

#: Keys of the representation specification.
REPRESENTATIONS_KEY = "representations"
KEEP_ORIGINAL_KEY = "keep_original"

#: Columns that are metadata and must never be used as predictors.
DEFAULT_METADATA_COLUMNS: tuple[str, ...] = (
    "participant_id",
    "session_id",
    "datetime",
    "file_id",
    "relative_path",
    "modality",
    "segment_id",
    "start_sample",
    "duration",
    "quality_indicator",
    TARGET_COLUMN,
)

#: Prefix used by every column created by this module.
REPR_PREFIX = "repr_"


# ======================================================================
# Ratio rules
# ======================================================================
@dataclass(frozen=True)
class RatioRule:
    """
    Definition of a single dimensionless ratio.

    The resulting column is::

        value = multiplier * prod(numerator) / prod(denominator)

    All the candidate names are looked up in the dataframe, and the rule is
    reported as skipped when a required column does not exist. This keeps the
    same code valid for PPG-only datasets (IEB-1, All-PPGs) and for the
    multimodal ones (IEB-2, IEB-3).

    Attributes
    ----------
    name:
        Name of the created column (without the ``repr_`` prefix).
    description:
        Short physiological/statistical description of the ratio.
    """

    name: str
    numerator: tuple[str, ...]
    denominator: tuple[str, ...]
    multiplier: float = 1.0
    description: str = ""


#: The catalogue of ratios. Every entry is dimensionless by construction.
RATIO_RULES: tuple[RatioRule, ...] = (
    RatioRule(
        name="ppg_beat_rate_hz",
        numerator=("ppg_b2_N_BEATS", "fs"),
        denominator=("duration",),
        description=(
            "Real heart rate: number of detected beats per second of segment. "
            "Independent of the segment duration and of the sensor gain."
        ),
    ),
    RatioRule(
        name="ppg_b2_width_over_period",
        numerator=("ppg_b2_width_median",),
        denominator=("ppg_b2_T_median",),
        description=(
            "Relative width of the pulse wave: fraction of the cardiac cycle "
            "spent in systole. Dimensionless, independent of the sensor gain "
            "and of the peripheral perfusion."
        ),
    ),
    RatioRule(
        name="ppg_b2_crest_time_over_period",
        numerator=("ppg_b2_Tr_median",),
        denominator=("ppg_b2_T_median",),
        description=(
            "Normalized crest time: fraction of the cardiac cycle spent until "
            "the systolic peak. Related to the systolic rise time and to the "
            "arterial stiffness."
        ),
    ),
    RatioRule(
        name="ppg_p1_rr_cv",
        numerator=("ppg_p1_peak_interval_std", "ppg_p1_hr_mean"),
        denominator=(),
        multiplier=1.0 / 60.0,
        description=(
            "Coefficient of variation of the pulse-to-pulse interval: "
            "std(RR) / mean(RR), with mean(RR) = 60 / hr_mean. A "
            "dimensionless measure of the pulse rate variability."
        ),
    ),
)


# ======================================================================
# Generic helpers
# ======================================================================
def detect_metadata_columns(
    df: pd.DataFrame,
    extra_columns: Sequence[str] = (),
) -> list[str]:
    """
    Return the metadata columns that are present in the dataframe.

    Parameters
    ----------
    df:
        Features dataframe.
    extra_columns:
        Additional columns that must be treated as metadata.

    Returns
    -------
    list of str
        Metadata column names, in a stable order.
    """
    found: list[str] = []
    for col in list(DEFAULT_METADATA_COLUMNS) + list(extra_columns):
        if col in df.columns and col not in found:
            found.append(col)
    return found


def select_feature_columns(
    df: pd.DataFrame,
    metadata_columns: Sequence[str] | None = None,
) -> list[str]:
    """
    Return the numeric columns of the dataframe that are actual features.

    Parameters
    ----------
    df:
        Features dataframe.
    metadata_columns:
        Metadata columns to exclude. Auto-detected when None.

    Returns
    -------
    list of str
        Feature column names, preserving the original order.
    """
    if metadata_columns is None:
        metadata_columns = detect_metadata_columns(df)
    blocked = set(metadata_columns)
    return [
        col
        for col in df.columns
        if col not in blocked and pd.api.types.is_numeric_dtype(df[col])
    ]


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    min_relative_denominator: float = 1e-3,
) -> pd.Series:
    """
    Element-wise ratio that maps an unusable denominator to NaN.

    A ratio is only meaningful when the denominator has a stable and non-zero
    scale. A denominator that is (almost) zero for some rows produces values
    of arbitrary magnitude, so it is replaced by NaN. The threshold is
    relative to the typical magnitude of the denominator, which makes the
    guard independent of the unit of the feature.
    """
    den = pd.to_numeric(denominator, errors="coerce")
    typical = float(np.nanmedian(np.abs(den.to_numpy(dtype=float)))) if den.notna().any() else 0.0
    threshold = min_relative_denominator * typical
    if np.isfinite(threshold) and threshold > 0:
        den = den.where(den.abs() > threshold, np.nan)
    den = den.where(den.abs() > EPS, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def _product_of_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    scalars: dict[str, float] | None = None,
) -> tuple[pd.Series, list[str]]:
    """
    Multiply the given columns and report the ones that are not available.

    A missing factor would silently change the meaning of a ratio, so the
    caller must skip the rule when the returned list is not empty.

    Parameters
    ----------
    df:
        Source dataframe.
    columns:
        Column names to multiply. A name present in ``scalars`` is replaced
        by the corresponding constant, which is how the sampling rate enters
        the expressions.
    scalars:
        Constant values available to the expression.

    Returns
    -------
    tuple
        ``(product_series, missing_column_names)``. The empty product is 1,
        and the product is all-NaN when none of the requested columns is
        available.
    """
    scalars = scalars or {}
    if not columns:
        return pd.Series(1.0, index=df.index, dtype=float), []
    result: pd.Series | None = None
    missing: list[str] = []
    for col in columns:
        if col in scalars:
            values = pd.Series(float(scalars[col]), index=df.index, dtype=float)
        elif col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
        else:
            missing.append(col)
            continue
        result = values if result is None else result * values
    if result is None:
        result = pd.Series(np.nan, index=df.index, dtype=float)
    return result, missing


def describe_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the original feature matrix.

    The summary is the starting point of any feature engineering work: it
    shows how many features exist, how many are usable, and how the samples
    are distributed among participants and glucose measurements.

    Parameters
    ----------
    df:
        Features dataframe.

    Returns
    -------
    pandas.DataFrame
        One row per feature with its basic statistics.
    """
    feature_columns = select_feature_columns(df)
    rows = []
    for col in feature_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "n_valid": int(series.notna().sum()),
                "pct_missing": float(100.0 * series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
                "mean": float(series.mean()) if series.notna().any() else np.nan,
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else np.nan,
                "min": float(series.min()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_ratio_features(
    df: pd.DataFrame,
    rules: Sequence[RatioRule] = RATIO_RULES,
    sampling_rate: float | None = None,
) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    """
    Build dimensionless ratios from the original features.

    The idea: an absolute value (amplitude in ADC units, power in dB^2/Hz,
    time in seconds) depends on the sensor, on the contact pressure and on
    the acquisition session. A ratio between two compatible quantities is
    invariant to those nuisance factors and is therefore much more likely to
    carry physiological information.

    Parameters
    ----------
    df:
        Features dataframe (one row per segment).
    rules:
        Catalogue of ratios to apply.
    sampling_rate:
        Sampling frequency in Hz. It is required by the rules that convert
        the ``duration`` column (expressed in samples) into seconds. The
        rules that need it are skipped when the value is None.

    Returns
    -------
    tuple
        ``(ratio_dataframe, applied_rule_names, skipped_rule_details)``.
        Each skipped entry is a dict with ``rule``, ``reason`` and ``missing``.
    """
    if df.empty:
        return pd.DataFrame(index=df.index), [], []

    scalars: dict[str, float] = {}
    if sampling_rate is not None:
        scalars[SAMPLING_RATE_KEY] = float(sampling_rate)

    out = pd.DataFrame(index=df.index)
    applied: list[str] = []
    skipped: list[dict[str, Any]] = []

    for rule in rules:
        numerator, missing_numerator = _product_of_columns(
            df, rule.numerator, scalars
        )
        denominator, missing_denominator = _product_of_columns(
            df, rule.denominator, scalars
        )
        missing = list(dict.fromkeys(missing_numerator + missing_denominator))
        if missing:
            skipped.append(
                {
                    "rule": rule.name,
                    "reason": "required columns are missing",
                    "missing": missing,
                }
            )
            continue
        if numerator.isna().all() or denominator.isna().all():
            skipped.append(
                {
                    "rule": rule.name,
                    "reason": "all values are missing",
                    "missing": [],
                }
            )
            continue
        values = rule.multiplier * _safe_ratio(numerator, denominator)
        out[REPR_PREFIX + rule.name] = values
        applied.append(rule.name)

    return out, applied, skipped


# ======================================================================
# Representation registry
# ======================================================================
RatioBuilder = Callable[
    ..., tuple[pd.DataFrame, list[str], list[dict[str, Any]]]
]

#: Name -> builder. Adding a new representation is a one-line change.
REPRESENTATIONS: dict[str, RatioBuilder] = {
    "ratio": build_ratio_features,
}

AVAILABLE_REPRESENTATIONS = tuple(REPRESENTATIONS)

DEFAULT_SPEC: dict[str, Any] = {
    REPRESENTATIONS_KEY: ["ratio"],
    KEEP_ORIGINAL_KEY: True,
}


def apply_representation(
    df: pd.DataFrame,
    spec: dict[str, Any] | None = None,
    sampling_rate: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply the requested representations to the original features.

    Parameters
    ----------
    df:
        Original features dataframe (one row per segment).
    spec:
        Specification of the representations. See ``DEFAULT_SPEC``.
    sampling_rate:
        Sampling frequency in Hz, used by the rules that convert the
        ``duration`` column into seconds.

    Returns
    -------
    tuple
        ``(result_dataframe, report)``.

    Notes
    -----
    When ``keep_original`` is false, the metadata columns (identifiers and
    target) are kept so that the output can still be joined with the labels.
    """
    spec = _merge_spec(spec)
    requested = list(spec.get(REPRESENTATIONS_KEY, []))

    unknown = [name for name in requested if name not in REPRESENTATIONS]
    if unknown:
        raise ValueError(
            f"Unknown representation(s): {unknown}. "
            f"Valid options: {AVAILABLE_REPRESENTATIONS}"
        )

    report: dict[str, Any] = {
        "requested": requested,
        "applied": [],
        "skipped": [],
        "n_input_rows": int(df.shape[0]),
        "n_input_features": len(select_feature_columns(df)),
    }

    created: list[pd.DataFrame] = []
    for name in requested:
        block, applied, skipped = REPRESENTATIONS[name](
            df, sampling_rate=sampling_rate
        )
        if applied:
            created.append(block)
            report["applied"].append(
                {"name": name, "n_columns": len(applied), "columns": applied}
            )
        for entry in skipped:
            report["skipped"].append({"name": name, **entry})

    if not created:
        return df.copy(), report

    representation = pd.concat(created, axis=1)
    keep_original = bool(spec.get(KEEP_ORIGINAL_KEY, True))
    if keep_original:
        representation = representation.loc[
            :, ~representation.columns.isin(df.columns)
        ]
        result = pd.concat([df, representation], axis=1)
    else:
        metadata = detect_metadata_columns(df)
        result = pd.concat([df[metadata], representation], axis=1)

    report["n_output_rows"] = int(result.shape[0])
    report["n_output_features"] = len(select_feature_columns(result))
    report["new_columns"] = [c for c in result.columns if c not in df.columns]
    return result, report


def _merge_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Merge the user specification with the default one."""
    merged = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in DEFAULT_SPEC.items()
    }
    if not spec:
        return merged
    for key, value in spec.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged
