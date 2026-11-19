from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from ..errors import BrokerUpstreamError, UpstreamSchemaError
from ..service import RiskService
from .fakes import FakeSource, account


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def test_all_account_cache_populates_specific_entries():
    source = FakeSource()
    service = RiskService(source, cache_ttl_seconds=2)

    first = service.accounts()
    specific = service.account("FAKE-0001")

    assert first.cache == "miss"
    assert specific.cache == "hit"
    assert source.all_calls == 1
    assert source.one_calls == []


def test_specific_refresh_uses_only_specific_account_call():
    source = FakeSource()
    service = RiskService(source)

    result = service.account("FAKE-7777", refresh=True)

    assert result.cache == "bypass"
    assert result.data.account_id == "FAKE-7777"
    assert source.one_calls == ["FAKE-7777"]
    assert source.all_calls == 0


def test_positions_route_selection_and_flattening_preserve_account_ids():
    source = FakeSource(all_payload=[account("FAKE-A"), account("FAKE-B")])
    service = RiskService(source)

    all_positions = service.positions()
    one_positions = service.positions(account_id="FAKE-C", refresh=True)

    assert [item.account_id for item in all_positions.data] == [
        "FAKE-A",
        "FAKE-B",
    ]
    assert [item.account_id for item in one_positions.data] == ["FAKE-C"]
    assert source.all_calls == 1
    assert source.one_calls == ["FAKE-C"]


def test_include_positions_false_retains_structural_count_and_omits_rows():
    source = FakeSource()
    service = RiskService(source)
    account_summary = service.accounts(include_positions=False).data[0]

    assert account_summary.position_count == 1
    assert "positions" not in account_summary.model_dump()


def test_expiry_and_refresh_bypass_cache():
    source = FakeSource()
    clock = Clock()
    service = RiskService(source, cache_ttl_seconds=2, monotonic_fn=clock)

    assert service.accounts().cache == "miss"
    assert service.accounts().cache == "hit"
    clock.value += 2.01
    assert service.accounts().cache == "miss"
    assert service.accounts(refresh=True).cache == "bypass"
    assert source.all_calls == 3


def test_zero_ttl_disables_cache():
    source = FakeSource()
    service = RiskService(source, cache_ttl_seconds=0)

    assert service.accounts().cache == "bypass"
    assert service.accounts().cache == "bypass"
    assert source.all_calls == 2


def test_malformed_response_is_not_cached():
    source = FakeSource(all_payload={"malformed": True})
    service = RiskService(source)

    with pytest.raises(UpstreamSchemaError):
        service.accounts()
    source.all_payload = [account()]
    assert service.accounts().data[0].account_id == "FAKE-0001"
    assert source.all_calls == 2


@pytest.mark.parametrize("failed_payload", [None, 401, 500, 503])
def test_failed_upstream_results_are_not_schema_or_not_found(failed_payload):
    source = FakeSource(all_payload=failed_payload)
    service = RiskService(source)

    with pytest.raises(BrokerUpstreamError):
        service.accounts()

    source = FakeSource(one_payload=failed_payload)
    service = RiskService(source)
    with pytest.raises(BrokerUpstreamError):
        service.account("FAKE-1")


def test_cached_models_are_returned_as_independent_objects():
    source = FakeSource()
    service = RiskService(source)

    first = service.accounts().data
    second = service.accounts().data

    assert first is not second
    assert first[0] is not second[0]
    assert first[0].positions is not second[0].positions


def test_concurrent_cache_misses_make_one_serialized_source_call():
    source = FakeSource(delay_seconds=0.05)
    service = RiskService(source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: service.accounts(), range(8)))

    assert source.all_calls == 1
    assert source.maximum_active_calls == 1
    assert sum(result.cache == "miss" for result in results) == 1
    assert sum(result.cache == "hit" for result in results) == 7
