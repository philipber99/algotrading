SET time_zone = '+00:00';

CREATE TABLE IF NOT EXISTS risk_option_price_capture_runs (
    run_id CHAR(36) NOT NULL,
    captured_at DATETIME(6) NOT NULL,
    status VARCHAR(32) NOT NULL,
    account_count INT UNSIGNED NOT NULL DEFAULT 0,
    position_count INT UNSIGNED NOT NULL DEFAULT 0,
    quote_count INT UNSIGNED NOT NULL DEFAULT 0,
    missing_count INT UNSIGNED NOT NULL DEFAULT 0,
    provider_class VARCHAR(255) NOT NULL,
    error_code VARCHAR(64) NULL,
    error_message VARCHAR(512) NULL,
    started_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    finished_at DATETIME(6) NULL,
    PRIMARY KEY (run_id),
    KEY ix_risk_option_capture_runs_time (captured_at),
    KEY ix_risk_option_capture_runs_status (status, captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS risk_option_position_prices (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    run_id CHAR(36) NOT NULL,
    account_id VARCHAR(128) NOT NULL,
    canonical_occ_symbol VARCHAR(32) NOT NULL,
    provider_option_symbol VARCHAR(32) NOT NULL,
    underlying_symbol VARCHAR(16) NOT NULL,
    expiration_date DATE NOT NULL,
    put_call VARCHAR(4) NOT NULL,
    strike DECIMAL(18,6) NOT NULL,
    signed_quantity DECIMAL(28,10) NOT NULL,
    captured_at DATETIME(6) NOT NULL,
    session_date DATE NOT NULL,
    quote_time DATETIME(6) NULL,
    trade_time DATETIME(6) NULL,
    bid DECIMAL(28,10) NULL,
    ask DECIMAL(28,10) NULL,
    mark DECIMAL(28,10) NOT NULL,
    last DECIMAL(28,10) NULL,
    open_price DECIMAL(28,10) NULL,
    high_price DECIMAL(28,10) NULL,
    low_price DECIMAL(28,10) NULL,
    close_price DECIMAL(28,10) NULL,
    underlying_price DECIMAL(28,10) NULL,
    provider_iv DECIMAL(18,10) NULL,
    provider_delta DECIMAL(18,10) NULL,
    provider_gamma DECIMAL(18,10) NULL,
    provider_theta_per_day DECIMAL(28,10) NULL,
    provider_vega_per_1pct DECIMAL(28,10) NULL,
    provider_rho_per_1pct DECIMAL(28,10) NULL,
    open_interest DECIMAL(28,10) NULL,
    volume DECIMAL(28,10) NULL,
    source VARCHAR(128) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_risk_option_position_price_session
        (account_id, canonical_occ_symbol, session_date),
    KEY ix_risk_option_position_price_symbol_session
        (canonical_occ_symbol, session_date, captured_at),
    KEY ix_risk_option_position_price_account_session
        (account_id, session_date, canonical_occ_symbol),
    KEY ix_risk_option_position_price_expiration
        (expiration_date, canonical_occ_symbol),
    CONSTRAINT fk_risk_option_position_price_run
        FOREIGN KEY (run_id) REFERENCES risk_option_price_capture_runs(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
