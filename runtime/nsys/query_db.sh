# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The script of querying the SQLite database.


# PCIe RX Throughput
sqlite3 ./db/tmp_db.sqlite "SELECT rawTimestamp, CAST(JSON_EXTRACT(data, '$.\"PCIe RX Throughput\"') as INTEGER) as value FROM GENERIC_EVENTS WHERE value > 0" > ./tmp/pcie_thr.txt

# SM Active
sqlite3 ./db/tmp_db.sqlite "SELECT rawTimestamp, CAST(JSON_EXTRACT(data, '$.\"SM Active\"') as INTEGER) as value FROM GENERIC_EVENTS WHERE value >= 0" > ./tmp/sm_active.txt

