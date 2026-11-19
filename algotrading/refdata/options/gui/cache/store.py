"""Immutable Parquet storage, active pointer, locking, and retention."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import CACHE_SCHEMA_VERSION, CacheConfig
from .errors import PublicationError, ValidationError
from .model import canonical_json, schema_fingerprint, sha256_file
from .validation import expected_schema, validate_parquet_file


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{utc_now():%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class WrittenFile:
    dataset: str
    staged_path: Path
    final_relative_path: str
    row_count: int
    byte_size: int
    sha256: str
    schema_fingerprint: str
    symbol: str | None = None
    minimum_session: str | None = None
    maximum_session: str | None = None

    def manifest_record(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "path": self.final_relative_path,
            "row_count": self.row_count,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "schema_fingerprint": self.schema_fingerprint,
            "symbol": self.symbol,
            "minimum_session": self.minimum_session,
            "maximum_session": self.maximum_session,
        }


class BuilderLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "BuilderLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise PublicationError(
                f"Another Time-of-Day cache builder holds {self.path}"
            ) from exc
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()} started={utc_now().isoformat()}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class CacheStore:
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.root = config.output_directory
        self.active_path = self.root / "active.json"
        self.manifests = self.root / "manifests"
        self.data = self.root / "data"
        self.staging = self.root / "staging"
        self.locks = self.root / "locks"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.manifests, self.data, self.staging, self.locks):
            path.mkdir(parents=True, exist_ok=True)

    def lock(self) -> BuilderLock:
        self.initialize()
        return BuilderLock(self.locks / "builder.lock")

    def create_staging(self, build_id: str) -> Path:
        self.initialize()
        path = self.staging / build_id
        path.mkdir(parents=False, exist_ok=False)
        return path

    def load_active_manifest(self, *, validate_files: bool = False) -> dict[str, Any] | None:
        if not self.active_path.exists():
            return None
        try:
            active = json.loads(self.active_path.read_text(encoding="utf-8"))
            generation_id = str(active["generation_id"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValidationError(f"Invalid active cache pointer: {exc}") from exc
        manifest = self.load_manifest(generation_id)
        if validate_files:
            self.validate_manifest_files(manifest)
        return manifest

    def load_manifest(self, generation_id: str) -> dict[str, Any]:
        if not generation_id or Path(generation_id).name != generation_id:
            raise ValidationError("Invalid cache generation identifier")
        path = self.manifests / f"{generation_id}.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Unable to read cache manifest {path}: {exc}") from exc
        if manifest.get("generation_id") != generation_id:
            raise ValidationError("Manifest generation identifier mismatch")
        if int(manifest.get("cache_schema_version", -1)) != CACHE_SCHEMA_VERSION:
            raise ValidationError(
                f"Unsupported cache schema version "
                f"{manifest.get('cache_schema_version')!r}"
            )
        return manifest

    def write_parquet(
        self,
        staging_root: Path,
        *,
        dataset: str,
        relative_path: str,
        table: pa.Table,
        symbol: str | None = None,
        session_values: Sequence[Any] = (),
    ) -> WrittenFile:
        final_relative = self._safe_relative(relative_path)
        if not final_relative.startswith("data/"):
            raise PublicationError("Published Parquet paths must be beneath data/")
        path = staging_root / final_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "compression": self.config.parquet_compression,
            "use_dictionary": True,
            "write_statistics": True,
            "row_group_size": 65536,
        }
        if self.config.parquet_compression in {"zstd", "gzip", "brotli"}:
            kwargs["compression_level"] = self.config.parquet_compression_level
        pq.write_table(table, path, **kwargs)
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())
        dates = sorted(str(value) for value in set(session_values))
        return WrittenFile(
            dataset=dataset,
            staged_path=path,
            final_relative_path=final_relative,
            row_count=table.num_rows,
            byte_size=path.stat().st_size,
            sha256=sha256_file(path),
            schema_fingerprint=schema_fingerprint(table.schema),
            symbol=symbol,
            minimum_session=dates[0] if dates else None,
            maximum_session=dates[-1] if dates else None,
        )

    def write_parquet_batches(
        self,
        staging_root: Path,
        *,
        dataset: str,
        relative_path: str,
        tables: Iterable[pa.Table],
        symbol: str | None = None,
        session_values: Sequence[Any] = (),
    ) -> WrittenFile:
        """Write an iterable of bounded Arrow batches to one Parquet file."""

        final_relative = self._safe_relative(relative_path)
        if not final_relative.startswith("data/"):
            raise PublicationError("Published Parquet paths must be beneath data/")
        path = staging_root / final_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = expected_schema(dataset)
        writer_kwargs: dict[str, Any] = {
            "compression": self.config.parquet_compression,
            "use_dictionary": True,
            "write_statistics": True,
        }
        if self.config.parquet_compression in {"zstd", "gzip", "brotli"}:
            writer_kwargs["compression_level"] = self.config.parquet_compression_level
        row_count = 0
        buffered_rows = 0
        buffered: list[pa.Table] = []
        writer = pq.ParquetWriter(path, schema, **writer_kwargs)

        def flush() -> None:
            nonlocal buffered_rows
            if not buffered_rows:
                return
            combined = pa.concat_tables(buffered) if len(buffered) > 1 else buffered[0]
            writer.write_table(combined, row_group_size=65536)
            buffered.clear()
            buffered_rows = 0

        try:
            for table in tables:
                if not table.schema.equals(schema, check_metadata=False):
                    raise PublicationError(
                        f"Arrow batch schema mismatch for {relative_path}"
                    )
                if table.num_rows:
                    buffered.append(table)
                    buffered_rows += table.num_rows
                    row_count += table.num_rows
                    if buffered_rows >= 49152:
                        flush()
            flush()
        finally:
            writer.close()
        if row_count == 0:
            raise PublicationError(f"Refusing to publish empty Parquet file {path}")
        with path.open("rb+") as handle:
            os.fsync(handle.fileno())
        dates = sorted(str(value) for value in set(session_values))
        return WrittenFile(
            dataset=dataset,
            staged_path=path,
            final_relative_path=final_relative,
            row_count=row_count,
            byte_size=path.stat().st_size,
            sha256=sha256_file(path),
            schema_fingerprint=schema_fingerprint(schema),
            symbol=symbol,
            minimum_session=dates[0] if dates else None,
            maximum_session=dates[-1] if dates else None,
        )

    def publish(
        self,
        staging_root: Path,
        manifest: Mapping[str, Any],
        written_files: Sequence[WrittenFile],
    ) -> dict[str, Any]:
        generation_id = str(manifest["generation_id"])
        if Path(generation_id).name != generation_id:
            raise PublicationError("Unsafe generation identifier")
        candidate = dict(manifest)
        candidate["cache_schema_version"] = CACHE_SCHEMA_VERSION
        inherited = [dict(item) for item in candidate.get("files", ())]
        new_records = [item.manifest_record() for item in written_files]
        paths = [str(item["path"]) for item in (*inherited, *new_records)]
        if len(paths) != len(set(paths)):
            raise PublicationError("Candidate manifest contains duplicate file paths")
        candidate["files"] = inherited + new_records
        candidate_path = staging_root / f"{generation_id}.json"
        self._write_json_fsync(candidate_path, candidate)

        for item in written_files:
            validate_parquet_file(
                item.staged_path,
                expected_schema=expected_schema(item.dataset),
                expected_rows=item.row_count,
                expected_sha256=item.sha256,
                compute_sha256=False,
            )

        # Every data move happens before the manifest/pointer switch. A crash
        # can leave only unreferenced immutable files, never a partial active
        # generation.
        moved_directories: set[Path] = set()
        for item in written_files:
            destination = self.root / item.final_relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise PublicationError(f"Immutable cache file already exists: {destination}")
            os.replace(item.staged_path, destination)
            moved_directories.add(destination.parent)
        for directory in moved_directories:
            self._fsync_directory(directory)
        manifest_path = self.manifests / f"{generation_id}.json"
        os.replace(candidate_path, manifest_path)
        self._fsync_directory(self.manifests)

        pointer = {
            "generation_id": generation_id,
            "manifest": f"manifests/{generation_id}.json",
            "activated_at_utc": utc_now().isoformat(),
        }
        pointer_tmp = self.root / f".active.{uuid.uuid4().hex}.tmp"
        self._write_json_fsync(pointer_tmp, pointer)
        os.replace(pointer_tmp, self.active_path)
        self._fsync_directory(self.root)
        with contextlib.suppress(OSError):
            shutil.rmtree(staging_root)
        return candidate

    def validate_manifest_files(self, manifest: Mapping[str, Any]) -> None:
        seen: set[str] = set()
        for record in manifest.get("files", ()):
            relative = self._safe_relative(str(record["path"]))
            if relative in seen:
                raise ValidationError(f"Manifest repeats file {relative}")
            seen.add(relative)
            path = self.root / relative
            validate_parquet_file(
                path,
                expected_schema=expected_schema(str(record["dataset"])),
                expected_rows=int(record["row_count"]),
                expected_sha256=str(record["sha256"]),
            )

    def referenced_files(self, manifest: Mapping[str, Any]) -> tuple[Path, ...]:
        return tuple(
            self.root / self._safe_relative(str(record["path"]))
            for record in manifest.get("files", ())
        )

    def cleanup(self, *, dry_run: bool = True) -> dict[str, Any]:
        self.initialize()
        retained_manifests = self._retained_manifests()
        referenced = {
            path.resolve()
            for manifest in retained_manifests
            for path in self.referenced_files(manifest)
        }
        cutoff = time.time() - self.config.staging_grace_hours * 3600
        removable_staging = [
            path
            for path in self.staging.iterdir()
            if path.stat().st_mtime < cutoff
        ]
        removable_data = [
            path
            for path in self.data.rglob("*.parquet")
            if path.resolve() not in referenced and path.stat().st_mtime < cutoff
        ]
        reclaimable_bytes = sum(path.stat().st_size for path in removable_data)
        if not dry_run:
            for path in removable_staging:
                shutil.rmtree(path)
            for path in removable_data:
                path.unlink()
            self._prune_empty_directories(self.data)
        return {
            "dry_run": dry_run,
            "staging_directories": [str(path) for path in removable_staging],
            "unreferenced_files": [str(path) for path in removable_data],
            "reclaimable_bytes": reclaimable_bytes,
        }

    def _retained_manifests(self) -> list[dict[str, Any]]:
        active = self.load_active_manifest()
        paths = sorted(
            self.manifests.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        retained: list[dict[str, Any]] = []
        if active is not None:
            retained.append(active)
        for path in paths:
            if len(retained) >= self.config.keep_previous_generations + 1:
                break
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not any(
                item.get("generation_id") == candidate.get("generation_id")
                for item in retained
            ):
                retained.append(candidate)
        return retained

    @staticmethod
    def _safe_relative(value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValidationError(f"Unsafe cache-relative path {value!r}")
        return path.as_posix()

    @staticmethod
    def _write_json_fsync(path: Path, value: Mapping[str, Any]) -> None:
        payload = canonical_json(dict(value)) + b"\n"
        with path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _prune_empty_directories(root: Path) -> None:
        for path in sorted(
            (item for item in root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            with contextlib.suppress(OSError):
                path.rmdir()
