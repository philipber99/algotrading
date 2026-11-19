"""Prospective option-chain definition identity and activation resolution."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Optional, Sequence

import pyarrow as pa

from algotrading.refdata.options.Common import (
    _parse_timestamp,
    _rows_to_arrow,
    validate_database_identifier,
)


UTC = timezone.utc


class DefinitionType(StrEnum):
    COLLECTION = "collection"
    NORMALIZATION = "normalization"
    CALCULATION = "calculation"
    SURFACE = "surface"


def canonical_configuration_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest32(value: str | bytes) -> bytes:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class Definition:
    definition_id: int
    definition_type: DefinitionType
    parent_definition_id: Optional[int]
    code_version: str
    configuration_hash: bytes
    solver_configuration_hash: bytes
    configuration_json: str
    activated_at: datetime
    deactivated_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_active: bool

    def applies_at(self, source_at: datetime) -> bool:
        source_at = _aware_utc(source_at)
        return self.activated_at <= source_at and (
            self.deactivated_at is None or source_at < self.deactivated_at
        )


DEFINITION_SCHEMA = pa.schema(
    [
        pa.field("definition_id", pa.uint32(), nullable=False),
        pa.field("definition_type", pa.string(), nullable=False),
        pa.field("parent_definition_id", pa.uint32()),
        pa.field("code_version", pa.string(), nullable=False),
        pa.field("configuration_hash", pa.binary(32), nullable=False),
        pa.field("solver_configuration_hash", pa.binary(32), nullable=False),
        pa.field("configuration_json", pa.string(), nullable=False),
        pa.field("activated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("deactivated_at", pa.timestamp("us", tz="UTC")),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("is_active", pa.bool_(), nullable=False),
    ]
)


def _aware_utc(value: datetime) -> datetime:
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"Invalid definition timestamp: {value!r}")
    return parsed


def compact_definition_id(
    definition_type: DefinitionType | str,
    *,
    parent_definition_id: Optional[int],
    code_version: str,
    configuration_hash: bytes,
    solver_configuration_hash: bytes,
) -> int:
    """Return a deterministic nonzero UInt32 surrogate.

    The full hashes are always compared on read/insert.  This compact value is
    only a lookup key; any UInt32 collision fails closed.
    """
    kind = DefinitionType(definition_type)
    payload = b"\x1f".join(
        (
            kind.value.encode("ascii"),
            str(parent_definition_id or 0).encode("ascii"),
            code_version.encode("utf-8"),
            bytes(configuration_hash),
            bytes(solver_configuration_hash),
        )
    )
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    return value or 1


def build_definition(
    definition_type: DefinitionType | str,
    *,
    code_version: str,
    configuration: Mapping[str, Any],
    solver_configuration: Optional[Mapping[str, Any]] = None,
    parent_definition_id: Optional[int] = None,
    activated_at: datetime,
    deactivated_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Definition:
    kind = DefinitionType(definition_type)
    if kind is DefinitionType.SURFACE and parent_definition_id is None:
        raise ValueError("A surface definition requires a parent calculation definition")
    configuration_json = canonical_configuration_json(configuration)
    solver_json = canonical_configuration_json(solver_configuration or {})
    configuration_hash = digest32(configuration_json)
    solver_hash = digest32(solver_json)
    activated_at = _aware_utc(activated_at)
    deactivated_at = _aware_utc(deactivated_at) if deactivated_at else None
    if deactivated_at is not None and deactivated_at <= activated_at:
        raise ValueError("Definition deactivation must be after activation")
    created_at = _aware_utc(now or datetime.now(UTC))
    definition_id = compact_definition_id(
        kind,
        parent_definition_id=parent_definition_id,
        code_version=code_version,
        configuration_hash=configuration_hash,
        solver_configuration_hash=solver_hash,
    )
    return Definition(
        definition_id=definition_id,
        definition_type=kind,
        parent_definition_id=parent_definition_id,
        code_version=code_version,
        configuration_hash=configuration_hash,
        solver_configuration_hash=solver_hash,
        configuration_json=configuration_json,
        activated_at=activated_at,
        deactivated_at=deactivated_at,
        created_at=created_at,
        updated_at=created_at,
        is_active=True,
    )


def validate_activation_intervals(definitions: Sequence[Definition]) -> None:
    by_type: dict[DefinitionType, list[Definition]] = {}
    for definition in definitions:
        if definition.is_active:
            by_type.setdefault(definition.definition_type, []).append(definition)
    for kind, items in by_type.items():
        ordered = sorted(items, key=lambda item: (item.activated_at, item.definition_id))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.deactivated_at is None or current.activated_at < previous.deactivated_at:
                raise ValueError(
                    f"Overlapping {kind.value} definition activation intervals: "
                    f"{previous.definition_id} and {current.definition_id}"
                )


def resolve_definition(
    definitions: Sequence[Definition],
    definition_type: DefinitionType | str,
    source_collected_at: datetime,
    *,
    parent_definition_id: Optional[int] = None,
) -> Definition:
    kind = DefinitionType(definition_type)
    source_at = _aware_utc(source_collected_at)
    candidates = [
        item
        for item in definitions
        if item.is_active
        and item.definition_type is kind
        and item.parent_definition_id == parent_definition_id
        and item.applies_at(source_at)
    ]
    if len(candidates) != 1:
        raise LookupError(
            f"Expected one {kind.value} definition at {source_at.isoformat()} "
            f"for parent {parent_definition_id}, found {len(candidates)}"
        )
    return candidates[0]


class DefinitionRepository:
    def __init__(self, client: Any, database: str):
        self.client = client
        self.database = validate_database_identifier(database)

    @staticmethod
    def _from_row(row: Mapping[str, Any]) -> Definition:
        return Definition(
            definition_id=int(row["definition_id"]),
            definition_type=DefinitionType(str(row["definition_type"])),
            parent_definition_id=(
                int(row["parent_definition_id"])
                if row.get("parent_definition_id") is not None
                else None
            ),
            code_version=str(row["code_version"]),
            configuration_hash=bytes(row["configuration_hash"]),
            solver_configuration_hash=bytes(row["solver_configuration_hash"]),
            configuration_json=str(row["configuration_json"]),
            activated_at=_aware_utc(row["activated_at"]),
            deactivated_at=(
                _aware_utc(row["deactivated_at"])
                if row.get("deactivated_at") is not None
                else None
            ),
            created_at=_aware_utc(row["created_at"]),
            updated_at=_aware_utc(row["updated_at"]),
            is_active=bool(row["is_active"]),
        )

    def load(self, definition_type: Optional[DefinitionType | str] = None) -> list[Definition]:
        parameters: dict[str, Any] = {}
        where = ""
        if definition_type is not None:
            where = " WHERE definition_type = {definition_type:String}"
            parameters["definition_type"] = DefinitionType(definition_type).value
        table = self.client.query_arrow(
            f"SELECT definition_id, toString(definition_type) AS definition_type, "
            "parent_definition_id, code_version, configuration_hash, "
            "solver_configuration_hash, configuration_json, activated_at, "
            "deactivated_at, created_at, updated_at, is_active "
            f"FROM {self.database}.options_chain_definitions FINAL{where}",
            parameters=parameters,
        )
        return [self._from_row(row) for row in table.to_pylist()]

    def ensure(self, definition: Definition) -> Definition:
        all_definitions = self.load()
        definitions = [
            item
            for item in all_definitions
            if item.definition_type is definition.definition_type
        ]
        for stored in all_definitions:
            if stored.definition_id != definition.definition_id:
                continue
            if (
                stored.definition_type is not definition.definition_type
                or stored.code_version != definition.code_version
                or stored.parent_definition_id != definition.parent_definition_id
                or stored.configuration_hash != definition.configuration_hash
                or stored.solver_configuration_hash
                != definition.solver_configuration_hash
            ):
                raise RuntimeError(
                    f"Definition UInt32 collision for {definition.definition_id}"
                )
            return stored
        validate_activation_intervals([*definitions, definition])
        row = {
            **definition.__dict__,
            "definition_type": definition.definition_type.value,
        }
        self.client.insert_arrow(
            f"{self.database}.options_chain_definitions",
            _rows_to_arrow([row], DEFINITION_SCHEMA),
            settings={"insert_deduplication_token": f"definition:{definition.definition_id}"},
        )
        return definition

    def activate_prospective(self, definition: Definition) -> Definition:
        """Install a definition at an explicit boundary without rewriting history.

        An open predecessor is closed at the new activation time by a current
        replacement row.  Bounded predecessors and future definitions are never
        silently altered, so overlaps and backdated activation fail closed.
        """
        all_definitions = self.load()
        definitions = [
            item
            for item in all_definitions
            if item.definition_type is definition.definition_type
        ]
        for stored in all_definitions:
            if stored.definition_id == definition.definition_id:
                if (
                    stored.definition_type is not definition.definition_type
                    or stored.code_version != definition.code_version
                    or stored.parent_definition_id != definition.parent_definition_id
                    or stored.configuration_hash != definition.configuration_hash
                    or stored.solver_configuration_hash
                    != definition.solver_configuration_hash
                ):
                    raise RuntimeError(
                        f"Definition UInt32 collision for {definition.definition_id}"
                    )
                return stored
        # Activation intervals are global within a definition type.  In
        # particular, a surface definition whose parent calculation changes
        # must close the prior surface interval rather than leave two active
        # surface definitions covering the same source timestamp.
        peers = [item for item in definitions if item.is_active]
        open_predecessors = [
            item
            for item in peers
            if item.deactivated_at is None
            and item.activated_at < definition.activated_at
        ]
        if len(open_predecessors) > 1:
            raise ValueError("Definition history has multiple open predecessors")
        now = definition.updated_at
        rows: list[Definition] = []
        candidate_history = list(peers)
        if open_predecessors:
            predecessor = open_predecessors[0]
            closed = replace(
                predecessor,
                deactivated_at=definition.activated_at,
                updated_at=now,
            )
            candidate_history = [
                closed if item.definition_id == closed.definition_id else item
                for item in candidate_history
            ]
            rows.append(closed)
        candidate_history.append(definition)
        validate_activation_intervals(candidate_history)
        rows.append(definition)
        payload = [
            {**item.__dict__, "definition_type": item.definition_type.value}
            for item in rows
        ]
        self.client.insert_arrow(
            f"{self.database}.options_chain_definitions",
            _rows_to_arrow(payload, DEFINITION_SCHEMA),
            settings={
                "insert_deduplication_token": f"definition-activation:{definition.definition_id}"
            },
        )
        logging.getLogger(__name__).info(
            "definition_activation definition_type=%s definition_id=%s "
            "parent_definition_id=%s activated_at=%s closed_predecessor_id=%s",
            definition.definition_type.value,
            definition.definition_id,
            definition.parent_definition_id,
            definition.activated_at.isoformat(),
            open_predecessors[0].definition_id if open_predecessors else None,
        )
        return definition

    def resolve(
        self,
        definition_type: DefinitionType | str,
        source_collected_at: datetime,
        *,
        parent_definition_id: Optional[int] = None,
    ) -> Definition:
        return resolve_definition(
            self.load(definition_type),
            definition_type,
            source_collected_at,
            parent_definition_id=parent_definition_id,
        )


__all__ = [
    "DEFINITION_SCHEMA",
    "Definition",
    "DefinitionRepository",
    "DefinitionType",
    "build_definition",
    "canonical_configuration_json",
    "compact_definition_id",
    "digest32",
    "resolve_definition",
    "validate_activation_intervals",
]
