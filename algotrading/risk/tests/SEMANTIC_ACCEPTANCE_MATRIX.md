# Risk semantic acceptance matrix

This matrix is the release-gate index for the 45 original (`O`) and 51
follow-up (`N`) requirements. Backend paths are relative to `algo_pkg/algotrading`;
GUI paths are relative to the workspace. A requirement may share a deterministic fixture with
another row, but every identifier has an explicit automated evidence anchor.

| ID | Automated evidence anchor |
|---|---|
| O01 | `risk/tests/test_semantic_correctness_remediation.py::test_es_terms_notional_tick_and_variation_pnl_reconcile_without_nav_value` |
| O02 | `risk/tests/test_semantic_correctness_remediation.py::test_missing_future_terms_fail_closed_and_cannot_report_full_coverage` |
| O03 | `risk/tests/test_semantic_correctness_remediation.py::test_es_terms_notional_tick_and_variation_pnl_reconcile_without_nav_value` |
| O04 | `risk/tests/test_semantic_correctness_remediation.py::test_futures_execution_realized_and_current_pnl_use_identical_terms` |
| O05 | `risk/tests/test_second_implementation_audit.py::test_continuous_future_history_requires_explicit_roll_adjustment_contract` |
| O06 | `risk/tests/test_followup_option_history_acceptance.py::test_two_exact_marks_use_contract_iv_and_reconcile_signed_short_position` |
| O07 | `risk/tests/test_followup_option_history_acceptance.py::test_iv_failure_uses_captured_greek_bridge_and_preserves_exact_total` |
| O08 | `risk/tests/test_followup_option_history_acceptance.py::test_marks_only_are_observed_repricing_not_generic_model_residual` |
| O09 | `risk/tests/test_followup_option_history_acceptance.py::test_missing_or_single_exact_mark_never_fabricates_an_interval` |
| O10 | `risk/tests/test_followup_option_history_acceptance.py::test_sql_mark_overrides_remote_mark_but_remote_fields_fill_sql_gaps` |
| O11 | `risk/tests/test_followup_option_history_acceptance.py::test_observation_captured_after_analysis_boundary_is_not_used` |
| O12 | `risk/tests/test_semantic_correctness_remediation.py::test_adjusted_deliverable_is_used_and_occ_assumption_cannot_certify_payoff` |
| O13 | `risk/tests/test_followup_semantic_acceptance.py::test_dividend_american_call_fails_closed_before_expiry` |
| O14 | `risk/tests/test_followup_semantic_acceptance.py::test_am_and_pm_authoritative_utc_timestamps_survive_dst_offsets` |
| O15 | `risk/tests/test_followup_semantic_acceptance.py::test_expiry_path_assigns_shares_and_carries_post_expiry_pnl` |
| O16 | `risk/tests/test_semantic_correctness_remediation.py::test_typed_scenarios_do_not_apply_broad_equity_to_crude_or_hide_missing_zn_dv01` |
| O17 | `risk/tests/test_semantic_correctness_remediation.py::test_typed_scenarios_do_not_apply_broad_equity_to_crude_or_hide_missing_zn_dv01` |
| O18 | `risk/tests/test_semantic_correctness_remediation.py::test_sector_scenario_uses_classified_holding_and_reports_exact_contribution` |
| O19 | `risk/tests/test_followup_option_history_acceptance.py::test_delta_neutral_straddle_keeps_nonlinear_exact_mark_history` |
| O20 | `risk/tests/test_followup_option_history_acceptance.py::test_pnl_tail_factor_cluster_and_hedge_share_the_same_marks_only_interval` |
| O21 | `risk/tests/test_core_calculations.py::test_factor_exposure_is_pnl_for_one_percent_factor_move` |
| O22 | `risk/api/tests/test_hedge_lab_routes.py::test_candidates_are_bounded_deterministic_priced_and_directly_usable` |
| O23 | `risk/tests/test_followup_semantic_acceptance.py::test_cluster_zero_history_weight_is_scenario_only_and_low_overlap_is_not_used` |
| O24 | `risk/tests/test_semantic_correctness_remediation.py::test_weighted_es_uses_exact_fractional_mass_and_reconciling_weights` |
| O25 | `risk/tests/test_semantic_correctness_remediation.py::test_payoff_certifies_long_stock_and_reports_short_stock_unbounded` |
| O26 | `risk/tests/test_semantic_correctness_remediation.py::test_payoff_certifies_long_stock_and_reports_short_stock_unbounded` |
| O27 | `risk/tests/test_second_implementation_audit.py::test_payoff_certifies_independent_underlyings_separately_and_not_mixed_expiries` |
| O28 | `risk/tests/test_second_implementation_audit.py::test_payoff_certifies_independent_underlyings_separately_and_not_mixed_expiries` |
| O29 | `risk/tests/test_second_implementation_audit.py::test_delta_neutral_tail_objective_discovers_convex_and_collar_sizes` |
| O30 | `risk/api/tests/test_hedge_lab_routes.py::test_portfolio_scenario_scan_returns_priced_index_candidates` |
| O31 | `risk/api/tests/test_hedge_lab_routes.py::test_portfolio_candidates_follow_measured_fit_not_instrument_class_priority` |
| O32 | `risk/api/tests/test_hedge_lab_routes.py::test_portfolio_candidates_follow_measured_fit_not_instrument_class_priority` |
| O33 | `risk/tests/test_followup_semantic_acceptance.py::test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros` |
| O34 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| O35 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| O36 | `risk/tests/test_core_calculations.py` sector derivative coverage fixtures |
| O37 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| O38 | `risk/api/tests/test_hedge_lab_routes.py::test_objective_validation_and_option_greek_units` |
| O39 | `risk_gui/tests/rendered-html.test.mjs::coverage labels, scenario attribution, paths, and runtime contracts` |
| O40 | `risk_gui/tests/rendered-html.test.mjs::Hedge Analysis is isolated, live-data-backed, and degrades explicitly` |
| O41 | `risk/tests/test_tian_service.py::test_real_vectorized_tian_returns_explicit_units_and_method_for_multiple_tenors` |
| O42 | `risk/tests/test_tian_service.py::test_target_selection_missing_rate_expiry_and_cache_fingerprint` |
| O43 | `risk/tests/test_followup_semantic_acceptance.py::test_stale_numeric_option_mark_and_greeks_are_not_modeled` |
| O44 | `risk/tests/test_semantic_correctness_remediation.py::test_freshness_accepts_latest_completed_eod_but_rejects_stale_intraday` |
| O45 | `risk/tests/test_semantic_correctness_remediation.py::test_nonfinite_and_incoherent_numeric_inputs_are_rejected_before_aggregation` |

| ID | Automated evidence anchor |
|---|---|
| N01 | `risk/tests/test_followup_option_history_acceptance.py::test_two_exact_marks_use_contract_iv_and_reconcile_signed_short_position` |
| N02 | `risk/tests/test_followup_option_history_acceptance.py::test_iv_failure_uses_captured_greek_bridge_and_preserves_exact_total` |
| N03 | `risk/tests/test_followup_option_history_acceptance.py::test_marks_only_are_observed_repricing_not_generic_model_residual` |
| N04 | `risk/tests/test_followup_option_history_acceptance.py::test_missing_or_single_exact_mark_never_fabricates_an_interval` |
| N05 | `risk/tests/test_followup_option_history_acceptance.py::test_sql_mark_overrides_remote_mark_but_remote_fields_fill_sql_gaps` |
| N06 | `risk/tests/test_followup_option_history_acceptance.py::test_observation_captured_after_analysis_boundary_is_not_used` |
| N07 | `risk/tests/test_followup_option_history_acceptance.py::test_pnl_tail_factor_cluster_and_hedge_share_the_same_marks_only_interval` |
| N08 | `risk/tests/test_followup_option_history_acceptance.py::test_hedge_lab_requires_and_forwards_risk_sql_option_history_dependency` |
| N09 | `risk/tests/test_followup_option_history_acceptance.py::test_delta_neutral_straddle_keeps_nonlinear_exact_mark_history` |
| N10 | `risk/tests/test_followup_option_history_acceptance.py::test_pnl_tail_factor_cluster_and_hedge_share_the_same_marks_only_interval` |
| N11 | `risk/tests/test_followup_semantic_acceptance.py::test_stale_numeric_option_mark_and_greeks_are_not_modeled` |
| N12 | `risk/tests/test_semantic_correctness_remediation.py::test_freshness_accepts_latest_completed_eod_but_rejects_stale_intraday` |
| N13 | `risk/tests/test_followup_semantic_acceptance.py::test_tiny_premium_unmodeled_short_call_makes_delta_denominator_unknown` |
| N14 | `risk/tests/test_followup_semantic_acceptance.py::test_missing_gamma_does_not_change_complete_vega_coverage` |
| N15 | `risk/tests/test_followup_semantic_acceptance.py::test_scenario_lists_stale_required_position_as_unmodeled` |
| N16 | `risk/tests/test_followup_semantic_acceptance.py::test_hard_stale_snapshot_preserves_nested_risk_coverage` |
| N17 | `risk/tests/test_semantic_correctness_remediation.py::test_adjusted_deliverable_is_used_and_occ_assumption_cannot_certify_payoff` |
| N18 | `risk/tests/test_followup_semantic_acceptance.py::test_mixed_adjusted_deliverable_separates_premium_and_exercise_economics` |
| N19 | `risk/tests/test_followup_semantic_acceptance.py::test_unsupported_deliverable_fails_payoff_and_expiry_scenario_closed` |
| N20 | `risk/tests/test_followup_semantic_acceptance.py::test_european_option_is_explicitly_unsupported_by_american_engine` |
| N21 | `risk/tests/test_second_implementation_audit.py::test_unknown_style_skips_local_iv_and_preserves_exact_observed_repricing` |
| N22 | `risk/tests/test_followup_semantic_acceptance.py::test_am_and_pm_authoritative_utc_timestamps_survive_dst_offsets` |
| N23 | `risk/tests/test_followup_semantic_acceptance.py::test_dividend_american_call_fails_closed_before_expiry` |
| N24 | `risk/tests/test_followup_semantic_acceptance.py::test_expiry_path_assigns_shares_and_carries_post_expiry_pnl` |
| N25 | `risk/tests/test_followup_semantic_acceptance.py::test_expiry_crossing_without_path_is_unmodeled` |
| N26 | `risk/tests/test_followup_semantic_acceptance.py::test_bond_etf_does_not_receive_broad_equity_shock` |
| N27 | `risk/tests/test_followup_semantic_acceptance.py::test_etf_economic_mappings_use_only_their_typed_channel` |
| N28 | `risk/tests/test_followup_semantic_acceptance.py::test_scenario_contributions_reconcile_and_partition_every_position` |
| N29 | `risk_gui/tests/rendered-html.test.mjs::coverage labels, scenario attribution, paths, and runtime contracts` |
| N30 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| N31 | `risk/tests/test_semantic_correctness_remediation.py::test_low_weight_extreme_changes_var_and_es_from_unweighted_order_statistics` plus GUI contract |
| N32 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| N33 | `risk_gui/tests/rendered-html.test.mjs::authoritative risk values remain backend-owned` |
| N34 | `risk_gui/tests/rendered-html.test.mjs::coverage labels, scenario attribution, paths, and runtime contracts` |
| N35 | `risk/tests/test_followup_semantic_acceptance.py::test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros` |
| N36 | `risk/tests/test_second_implementation_audit.py::test_bounded_quantity_search_enumerates_every_admissible_integer` |
| N37 | `risk/tests/test_followup_semantic_acceptance.py::test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros` |
| N38 | `risk/tests/test_followup_semantic_acceptance.py::test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros` |
| N39 | `risk/tests/test_followup_semantic_acceptance.py::test_future_objective_mapping_excludes_unrelated_contracts_and_includes_micros` |
| N40 | `risk/api/tests/test_hedge_lab_routes.py::test_candidates_are_bounded_deterministic_priced_and_directly_usable` |
| N41 | `risk/tests/test_second_implementation_audit.py::test_delta_neutral_tail_objective_discovers_convex_and_collar_sizes` |
| N42 | `risk/tests/test_followup_option_history_acceptance.py::test_pnl_tail_factor_cluster_and_hedge_share_the_same_marks_only_interval` |
| N43 | `risk/api/tests/test_hedge_lab_routes.py::test_candidates_are_bounded_deterministic_priced_and_directly_usable` |
| N44 | `risk/api/tests/test_hedge_lab_routes.py::test_portfolio_candidates_follow_measured_fit_not_instrument_class_priority` |
| N45 | `risk/tests/test_second_implementation_audit.py::test_candidate_endpoint_fully_evaluates_integer_set_and_selects_objective_optimum` |
| N46 | `risk/api/tests/test_hedge_lab_routes.py::test_candidate_universe_reserves_capacity_for_liquid_market_proxies` |
| N47 | `risk/tests/test_followup_semantic_acceptance.py::test_cluster_zero_history_weight_is_scenario_only_and_low_overlap_is_not_used` |
| N48 | `risk/tests/test_followup_semantic_acceptance.py::test_cluster_zero_history_weight_is_scenario_only_and_low_overlap_is_not_used` |
| N49 | `risk/tests/test_followup_semantic_acceptance.py::test_holiday_early_close_and_futures_sessions_use_versioned_calendars` |
| N50 | `risk/tests/test_second_implementation_audit.py::test_cross_year_and_product_calendars_are_bounded_and_distinct` |
| N51 | `risk/tests/test_followup_semantic_acceptance.py::test_semantic_configuration_rejects_invalid_consumed_controls` and GUI runtime-contract test |

Second-audit regressions are concentrated in
`risk/tests/test_second_implementation_audit.py`; they cover partial terms,
unknown style, integer optimum/positive improvement, material nonlinear
omissions, delta-neutral convex/collar discovery, public paths, economic mapping
order, missing-field remote fill, and cross-year product calendars.
