"""Canonical OCC option-symbol parsing shared by live ingress routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import re


_ROOT = re.compile(r"^[A-Z0-9.\-]{1,6}$")
_SUFFIX = re.compile(r"^(?P<date>\d{6})(?P<put_call>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiration: date
    put_call: str
    strike: Decimal
    canonical: str
    provider: str


def parse_occ_symbol(value: str) -> OccSymbol:
    if not isinstance(value, str):
        raise ValueError("option symbol must be text")
    raw = value.upper().rstrip()
    if len(raw) < 16:
        raise ValueError("option symbol is too short")

    # The fixed-width suffix keeps root padding separate from date and strike parsing.
    suffix_text = raw[-15:]
    suffix = _SUFFIX.fullmatch(suffix_text)
    if suffix is None:
        raise ValueError("option symbol must end in YYMMDD, C/P, and 8 strike digits")
    root = raw[:-15].strip()
    if not _ROOT.fullmatch(root):
        raise ValueError("option root must contain 1-6 supported OCC characters")
    try:
        expiration = date(
            2000 + int(suffix.group("date")[0:2]),
            int(suffix.group("date")[2:4]),
            int(suffix.group("date")[4:6]),
        )
        # OCC encodes strikes as an eight-digit integer in thousandths.
        strike = Decimal(suffix.group("strike")) / Decimal(1000)
    except (ValueError, InvalidOperation) as exc:
        raise ValueError("option symbol contains an invalid date or strike") from exc
    canonical = f"{root}{suffix_text}"
    # Pad only the provider form; keep canonical contract IDs compact.
    provider = f"{root:<6}{suffix_text}"
    return OccSymbol(
        root=root,
        expiration=expiration,
        put_call="CALL" if suffix.group("put_call") == "C" else "PUT",
        strike=strike,
        canonical=canonical,
        provider=provider,
    )


def build_occ_symbol(
    root: str,
    expiration: date,
    put_call: str,
    strike: float | Decimal,
) -> OccSymbol:
    normalized_root = str(root).strip().upper()
    if not _ROOT.fullmatch(normalized_root):
        raise ValueError("invalid OCC root")
    side = str(put_call).strip().upper()
    if side not in {"CALL", "PUT", "C", "P"}:
        raise ValueError("put_call must be CALL or PUT")
    strike_decimal = Decimal(str(strike))
    encoded = strike_decimal * Decimal(1000)
    # Reject strikes that cannot round-trip through the exact OCC representation.
    if encoded != encoded.to_integral_value() or encoded < 0 or encoded > 99999999:
        raise ValueError("strike cannot be represented by OCC encoding")
    suffix = (
        expiration.strftime("%y%m%d")
        + ("C" if side in {"CALL", "C"} else "P")
        + f"{int(encoded):08d}"
    )
    return parse_occ_symbol(f"{normalized_root}{suffix}")


__all__ = ["OccSymbol", "build_occ_symbol", "parse_occ_symbol"]
