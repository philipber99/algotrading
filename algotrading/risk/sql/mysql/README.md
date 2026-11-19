# Risk transaction schema

Run the idempotent initializer against the already-created application database:

```bash
mysql --host "$RISK_DB_HOST" --user "$RISK_DB_USER" --password \
  "$RISK_DB_NAME" < algotrading/risk/sql/mysql/001_create_risk_transactions.sql

mysql --host "$RISK_DB_HOST" --user "$RISK_DB_USER" --password \
  "$RISK_DB_NAME" < algotrading/risk/sql/mysql/002_create_risk_option_prices.sql

mysql --host "$RISK_DB_HOST" --user "$RISK_DB_USER" --password \
  "$RISK_DB_NAME" < algotrading/risk/sql/mysql/003_expand_risk_option_daily_chain_capture.sql
```

The script never creates, selects, or drops a database. The MySQL session is
forced to UTC; application code converts aware UTC values to `DATETIME(6)` at
the repository boundary.

`risk_transaction_fees` stores each broker-reported component fee using
`transaction_id` as its foreign key to `risk_transactions`. Its primary key
starts with `transaction_id`, so `JOIN risk_transaction_fees fee ON
fee.transaction_id = transaction.id` is an indexed lookup.
Only non-zero component fees are persisted; fee transfer items are not stored
as transaction execution items.

`risk_transaction_sync_state` records the interval successfully checked by the
daily broker feed, including successful runs that return zero transactions.
Coverage therefore describes the feed run rather than incorrectly treating the
most recent transaction timestamp as a synchronization watermark. The importer
resumes from that successful watermark with its configured overlap.

The second script stores one authoritative captured mark per account, exact OCC
contract, and market session. Re-running the nightly capture for the same
session updates that row rather than duplicating it.

Run the third script once, after script `002`, to add the compact daily
underlying/expiration chain snapshots and their raw option surface points.
