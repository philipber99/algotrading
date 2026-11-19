-- Add compact daily option-chain observations to the existing Risk option-price
-- repository. Run this script once after 002_create_risk_option_prices.sql.
--
-- This is an additive MySQL upgrade only:
--   * no existing column is changed or removed;
--   * no historical data is updated or backfilled;
--   * existing risk_option_position_prices rows remain valid with NULL links.

SET time_zone = '+00:00';

CREATE TABLE risk_option_chain_snapshots (
    chain_snapshot_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_id CHAR(36) NOT NULL,
    underlying_symbol VARCHAR(16)
        CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    expiration_date DATE NOT NULL,
    spot DECIMAL(16,6) NULL,
    spot_source TINYINT UNSIGNED NULL COMMENT
        '1=response underlyingPrice, 2=underlying mark, 3=midpoint, 4=last',
    underlying_quote_time DATETIME(3) NULL,
    snapshot_status TINYINT UNSIGNED NOT NULL COMMENT
        '1=success, 2=partial, 3=failed',
    snapshot_flags TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT
        'Bitmask: 1=delayed, 2=truncated, 4=index, 8=symbol mismatch',
    error_code VARCHAR(32)
        CHARACTER SET ascii COLLATE ascii_bin NULL,
    PRIMARY KEY (chain_snapshot_id),
    UNIQUE KEY uq_risk_option_chain_snapshot_run_expiry
        (run_id, underlying_symbol, expiration_date),
    KEY ix_risk_option_chain_snapshot_history
        (underlying_symbol, expiration_date, run_id),
    CONSTRAINT fk_risk_option_chain_snapshot_run
        FOREIGN KEY (run_id)
        REFERENCES risk_option_price_capture_runs(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE risk_option_surface_points (
    chain_snapshot_id INT UNSIGNED NOT NULL,
    canonical_occ_symbol VARCHAR(21)
        CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    quote_time DATETIME(3) NULL,
    trade_time DATETIME(3) NULL,
    bid DECIMAL(16,6) NULL,
    ask DECIMAL(16,6) NULL,
    mark DECIMAL(16,6) NULL,
    last DECIMAL(16,6) NULL,
    open_interest INT UNSIGNED NULL,
    volume INT UNSIGNED NULL,
    observation_source TINYINT UNSIGNED NOT NULL COMMENT
        '1=option chain, 2=exact-option quote fallback',
    security_status TINYINT UNSIGNED NULL COMMENT
        '0=unknown, 1=normal, 2=halted, 3=closed or inactive',
    PRIMARY KEY (chain_snapshot_id, canonical_occ_symbol),
    KEY ix_risk_option_surface_contract_history
        (canonical_occ_symbol, chain_snapshot_id),
    CONSTRAINT fk_risk_option_surface_point_snapshot
        FOREIGN KEY (chain_snapshot_id)
        REFERENCES risk_option_chain_snapshots(chain_snapshot_id)
) ENGINE=InnoDB DEFAULT CHARSET=ascii COLLATE=ascii_bin;

ALTER TABLE risk_option_price_capture_runs
    ADD COLUMN chain_count INT UNSIGNED NOT NULL DEFAULT 0,
    ADD COLUMN surface_point_count INT UNSIGNED NOT NULL DEFAULT 0;

ALTER TABLE risk_option_position_prices
    ADD COLUMN chain_snapshot_id INT UNSIGNED NULL,
    ADD COLUMN mark_source TINYINT UNSIGNED NULL COMMENT
        '1=provider mark, 2=valid midpoint, 3=last, 4=close';
