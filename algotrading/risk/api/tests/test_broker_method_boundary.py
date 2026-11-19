from __future__ import annotations

import ast
import importlib
from pathlib import Path

from .. import broker_source
from ..broker_source import BrokerAccountSource, create_tda_broker


def test_get_accounts_requests_positions_in_aggregate_response(monkeypatch):
    broker_module = importlib.import_module("algotrading.tda.TDABrokerInterface")
    calls = []

    def fake_request(context, uri, parameters=None, **kwargs):
        calls.append((context, uri, parameters, kwargs))
        return []

    monkeypatch.setattr(broker_module, "make_request", fake_request)
    broker = object.__new__(broker_module.TDABrokerInterface)
    broker.context = {"test": True}

    assert broker.get_accounts() == []
    assert calls == [
        (
            broker.context,
            broker_module.accounts_uri,
            {"fields": "positions"},
            {},
        )
    ]


def test_real_broker_factory_passes_identical_context_and_explicit_false(monkeypatch):
    calls = []

    class RecordingBroker:
        def __init__(self, context, *, streaming):
            calls.append((context, streaming))

    monkeypatch.setattr(broker_source, "TDABrokerInterface", RecordingBroker)
    context = {"identity": object()}

    create_tda_broker(context)

    assert calls == [(context, False)]
    assert calls[0][0] is context


def test_broker_source_invokes_only_the_two_allowed_methods():
    class BoundaryBroker:
        def __init__(self):
            self.calls = []

        def get_accounts(self):
            self.calls.append(("get_accounts", None))
            return []

        def get_account(self, *, account):
            self.calls.append(("get_account", account))
            return 404

        def prohibited_helper(self):
            raise AssertionError("must not be called")

        def close(self):
            raise AssertionError("must not be called")

    broker = BoundaryBroker()
    source = BrokerAccountSource({}, broker_factory=lambda _: broker)

    source.fetch_all()
    source.fetch_one("FAKE-1")

    assert broker.calls == [
        ("get_accounts", None),
        ("get_account", "FAKE-1"),
    ]


def test_application_source_has_strict_static_broker_boundary():
    package_root = Path(__file__).resolve().parents[1]
    application_files = sorted(
        path for path in package_root.glob("*.py") if path.name != "__init__.py"
    )
    broker_calls = []
    constructors = []

    for path in application_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                else:
                    imported = [node.module or ""]
                    imported.extend(alias.name for alias in node.names)
                joined = " ".join(imported)
                assert "algotrading.tda.Session" not in joined
                assert "AccountActivityStreaming" not in joined
                assert "algotrading.tda.Streaming" not in joined

            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "TDABrokerInterface":
                constructors.append((path, node))
                streaming = next(
                    (kw.value for kw in node.keywords if kw.arg == "streaming"),
                    None,
                )
                assert isinstance(streaming, ast.Constant)
                assert streaming.value is False

            if isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if (
                    isinstance(receiver, ast.Attribute)
                    and isinstance(receiver.value, ast.Name)
                    and receiver.value.id == "self"
                    and receiver.attr == "_broker"
                ):
                    broker_calls.append((path, node.func.attr, node))
                    assert path.name == "broker_source.py"
                    assert node.func.attr in {"get_accounts", "get_account"}
                    if node.func.attr == "get_account":
                        assert [kw.arg for kw in node.keywords] == ["account"]

    assert len(constructors) == 1
    assert [name for _, name, _ in broker_calls] == [
        "get_accounts",
        "get_account",
    ]
