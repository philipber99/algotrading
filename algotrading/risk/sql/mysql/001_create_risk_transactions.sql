SET time_zone = '+00:00';

CREATE TABLE IF NOT EXISTS risk_transactions (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    account_id VARCHAR(128) NOT NULL,
    broker_transaction_id VARCHAR(191) NOT NULL,
    identifier_method VARCHAR(32) NOT NULL DEFAULT 'broker_id',
    transaction_type VARCHAR(64) NOT NULL,
    transaction_subtype VARCHAR(128) NULL,
    transaction_at DATETIME(6) NOT NULL,
    trade_date DATE NULL,
    settlement_date DATE NULL,
    symbol VARCHAR(64) NULL,
    asset_type VARCHAR(64) NULL,
    description VARCHAR(1024) NULL,
    amount DECIMAL(28,10) NULL,
    fees DECIMAL(28,10) NULL,
    net_amount DECIMAL(28,10) NULL,
    currency CHAR(3) NULL,
    order_id VARCHAR(191) NULL,
    source_updated_at DATETIME(6) NULL,
    ingested_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_risk_transactions_account_broker
        (account_id, broker_transaction_id),
    KEY ix_risk_transactions_account_time
        (account_id, transaction_at, id),
    KEY ix_risk_transactions_account_type_time
        (account_id, transaction_type, transaction_at, id),
    KEY ix_risk_transactions_account_symbol_time
        (account_id, symbol, transaction_at, id),
    KEY ix_risk_transactions_order_id (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS risk_transaction_items (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    transaction_id BIGINT UNSIGNED NOT NULL,
    item_index SMALLINT UNSIGNED NOT NULL,
    symbol VARCHAR(64) NULL,
    asset_type VARCHAR(64) NULL,
    instruction VARCHAR(64) NULL,
    position_effect VARCHAR(64) NULL,
    quantity DECIMAL(28,10) NULL,
    price DECIMAL(28,10) NULL,
    amount DECIMAL(28,10) NULL,
    cost DECIMAL(28,10) NULL,
    currency CHAR(3) NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
        ON UPDATE CURRENT_TIMESTAMP(6),
    PRIMARY KEY (id),
    UNIQUE KEY uq_risk_transaction_items_parent_index
        (transaction_id, item_index),
    KEY ix_risk_transaction_items_symbol (symbol, transaction_id),
    CONSTRAINT fk_risk_transaction_items_parent
        FOREIGN KEY (transaction_id) REFERENCES risk_transactions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS risk_transaction_fees (
    transaction_id BIGINT UNSIGNED NOT NULL,
    fee_index SMALLINT UNSIGNED NOT NULL,
    fee_type VARCHAR(64) NOT NULL,
    amount DECIMAL(28,10) NOT NULL,
    PRIMARY KEY (transaction_id, fee_index),
    KEY ix_risk_transaction_fees_type_transaction (fee_type, transaction_id),
    CONSTRAINT fk_risk_transaction_fees_parent
        FOREIGN KEY (transaction_id) REFERENCES risk_transactions(id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS risk_transaction_sync_state (
    account_id VARCHAR(128) NOT NULL,
    transaction_type VARCHAR(64) NOT NULL,
    first_synced_at DATETIME(6) NOT NULL,
    synced_through_at DATETIME(6) NOT NULL,
    last_success_at DATETIME(6) NOT NULL,
    PRIMARY KEY (account_id, transaction_type),
    KEY ix_risk_transaction_sync_state_success (last_success_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
