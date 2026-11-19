# Hybrid American surface delta nodes

American delta-node generation uses Tian as a fast proposal engine and the
configured American pricing method as the final authority. A node is published
only when the authoritative delta is finite and within
`options_calcs.surface.node_delta_tolerance` of its target.

Recommended production settings:

```text
options_calcs.surface.node_hybrid_enabled = true
options_calcs.surface.node_search_method = tian
options_calcs.surface.node_validation_method = configured
options_calcs.surface.node_tian_grid_size = 81
options_calcs.surface.node_validation_max_evaluations = 4
options_calcs.surface.node_validation_initial_candidates = 1
options_calcs.surface.node_legacy_exact_fallback_enabled = true
```

All authoritative calls count against the per-node validation budget, including
the initial candidate, bracket probes, and correction steps. The optional final
fallback is restricted to the Tian bracket and its nearest finite neighbors. It
does not construct a broad Crank–Nicolson grid. Set
`node_hybrid_enabled=false` and `node_search_method=observed_then_analytic` only
for temporary legacy A/B comparisons.

The worker logs `event=surface_node_methods` at startup and
`event=surface_numerical_summary` per completed surface. The latter contains
the Tian/authoritative evaluation counts, validation outcomes, fallback counts,
and all surface stage timings, including `surface.total_seconds`.

## Read-only ClickHouse A/B benchmark

The benchmark reads completed fit snapshots, runs both algorithms in memory,
and never publishes calculation or surface rows:

```bash
ALGO_HOME=/path/to/algo_pkg \
python -m algotrading.refdata.options.Benchmarks \
  -e production \
  --source clickhouse \
  --surface-node-ab \
  --sample-limit 40 \
  --symbols AMD,AMZN,INTC,MSFT,CVX \
  --output-dir tests/benchmark_output
```

Use `--snapshot-id UUID` to benchmark every selected expiration from one
surface, or `--from-date` and `--to-date` for a bounded deterministic window.
The CSV, JSON, and Markdown outputs include wall time, speedup, Tian and
authoritative evaluation counts, dense-grid counts, node availability, final
delta residuals, and strike differences versus the legacy path.

## Implementation validation sample

On 2026-07-19, a read-only shadow run against 18 stored expirations from one
AMD snapshot produced the following results after adapter warm-up:

- Legacy/hybrid successful delta nodes: 73 / 73.
- Legacy/hybrid authoritative evaluations: 3,530 / 223 (93.7% fewer).
- Legacy/hybrid broad dense evaluations: 3,161 / 0.
- Legacy/hybrid node wall time: 3.748 / 0.507 seconds (7.39x speedup).
- Maximum hybrid final-delta residual: `9.81e-6`.
- Maximum absolute strike difference versus legacy: `0.00631`.

This bounded result is evidence for rollout, not a guarantee for every symbol
or market regime. Re-run the command above on a larger representative window
and investigate any availability regression over 0.5 percentage points.
