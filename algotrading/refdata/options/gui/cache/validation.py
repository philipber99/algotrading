"""Publication gates for in-memory rows and immutable Parquet generations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .errors import ValidationError
from .model import (
    SESSION_INDEX_SCHEMA,
    SURFACE_OBSERVATION_SCHEMA,
    schema_fingerprint,
    sha256_file,
)


@dataclass
class ValidationReport:
    surface_rows: int = 0
    session_rows: int = 0
    rejected_rows: int = 0
    duplicate_keys: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface_rows": self.surface_rows,
            "session_rows": self.session_rows,
            "rejected_rows": self.rejected_rows,
            "duplicate_keys": self.duplicate_keys,
            "warnings": list(self.warnings),
            "rejection_reasons": dict(self.rejection_reasons),
        }


def _numpy_numeric(table: pa.Table, name: str) -> np.ndarray:
    column = table[name].combine_chunks()
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def validate_surface_table(
    table: pa.Table,
    *,
    max_rejected_fraction: float,
) -> tuple[pa.Table, ValidationReport]:
    """Validate a surface batch with column masks and preserve Arrow lists."""

    report = ValidationReport(surface_rows=table.num_rows)
    if table.num_rows == 0:
        return table, report
    missing = [name for name in SURFACE_OBSERVATION_SCHEMA.names if name not in table.column_names]
    if missing:
        raise ValidationError(f"Surface batch is missing columns: {missing}")

    size = table.num_rows
    valid = np.ones(size, dtype=bool)

    def reject(mask: np.ndarray, reason: str) -> None:
        selected = valid & mask
        count = int(np.count_nonzero(selected))
        if count:
            report.rejection_reasons[reason] += count
        valid[selected] = False

    for name in ("spot_price", "forward_price", "time_to_expiry_seconds"):
        values = _numpy_numeric(table, name)
        reject(~np.isfinite(values) | (values <= 0.0), f"invalid_{name}")
    for name in (
        "risk_free_rate",
        "dividend_yield",
        "weighted_rmse",
        "maximum_absolute_error",
    ):
        values = _numpy_numeric(table, name)
        invalid = ~np.isfinite(values)
        if name in {"weighted_rmse", "maximum_absolute_error"}:
            invalid |= values < 0.0
        reject(invalid, f"invalid_{name}")

    minimum = _numpy_numeric(table, "minimum_log_forward_moneyness")
    maximum = _numpy_numeric(table, "maximum_log_forward_moneyness")
    reject(
        ~np.isfinite(minimum) | ~np.isfinite(maximum) | (minimum >= maximum),
        "invalid_fit_coverage",
    )

    methods = np.char.upper(
        np.char.strip(np.asarray(table["fit_method"].to_pylist(), dtype=str))
    )
    reject(~np.isin(methods, ("SVI", "PCHIP")), "invalid_fit_method")
    svi = valid & (methods == "SVI")
    if np.any(svi):
        parameter_valid = np.ones(size, dtype=bool)
        for name in ("svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma"):
            parameter_valid &= np.isfinite(_numpy_numeric(table, name))
        parameter_valid &= _numpy_numeric(table, "svi_b") >= 0.0
        parameter_valid &= np.abs(_numpy_numeric(table, "svi_rho")) < 1.0
        parameter_valid &= _numpy_numeric(table, "svi_sigma") > 0.0
        reject(svi & ~parameter_valid, "invalid_svi_parameters")

    pchip_k = table["pchip_k"]
    pchip_w = table["pchip_total_variance"]
    for index in np.flatnonzero(valid & (methods == "PCHIP")):
        knots = np.asarray(pchip_k[int(index)].as_py() or (), dtype=np.float64)
        variances = np.asarray(pchip_w[int(index)].as_py() or (), dtype=np.float64)
        if (
            len(knots) < 2
            or len(knots) != len(variances)
            or not np.all(np.isfinite(knots))
            or not np.all(np.isfinite(variances))
            or np.any(np.diff(knots) <= 0.0)
            or np.any(variances <= 0.0)
        ):
            reject(np.arange(size) == index, "invalid_pchip_parameters")

    report.rejected_rows = int((~valid).sum())
    rejected_fraction = report.rejected_rows / report.surface_rows
    if rejected_fraction > max_rejected_fraction:
        raise ValidationError(
            "Rejected surface-row fraction "
            f"{rejected_fraction:.4f} exceeds {max_rejected_fraction:.4f}; "
            f"reasons={dict(report.rejection_reasons)}"
        )

    candidate = table.filter(pa.array(valid))
    if candidate.num_rows:
        keys = [
            "underlying_symbol",
            "session_date",
            "session_minute",
            "expiration_at",
            "price_basis",
        ]
        grouped = candidate.select(keys).group_by(keys).aggregate([([], "count_all")])
        duplicate_counts = _numpy_numeric(grouped, "count_all")
        report.duplicate_keys = int(np.sum(duplicate_counts > 1.0))
        if report.duplicate_keys:
            raise ValidationError(
                f"Candidate contains {report.duplicate_keys} duplicate logical keys"
            )
    return candidate, report


def table_from_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> pa.Table:
    try:
        return pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise ValidationError(f"Rows do not match Parquet schema: {exc}") from exc


def validate_parquet_file(
    path: Path,
    *,
    expected_schema: pa.Schema,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    metadata = pq.read_metadata(path)
    schema = metadata.schema.to_arrow_schema()
    if not schema.equals(expected_schema, check_metadata=False):
        raise ValidationError(f"Parquet schema mismatch: {path}")
    if expected_rows is not None and metadata.num_rows != expected_rows:
        raise ValidationError(
            f"Parquet row-count mismatch for {path}: "
            f"{metadata.num_rows} != {expected_rows}"
        )
    digest = sha256_file(path) if compute_sha256 else expected_sha256
    if compute_sha256 and expected_sha256 is not None and digest != expected_sha256:
        raise ValidationError(f"Parquet checksum mismatch: {path}")
    return {
        "rows": metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "schema_fingerprint": schema_fingerprint(schema),
    }


def expected_schema(dataset_name: str) -> pa.Schema:
    if dataset_name == "session_index":
        return SESSION_INDEX_SCHEMA
    if dataset_name == "surface_observations":
        return SURFACE_OBSERVATION_SCHEMA
    raise ValidationError(f"Unknown cache dataset {dataset_name!r}")
