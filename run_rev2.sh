#!/bin/bash
cd "$(dirname "$0")"
for e in e7 e4d; do
  echo "=== $e start $(date)" >> logs/run_rev.log
  python3 trident/run_experiments.py $e >> logs/${e}_v2.log 2>&1
  echo "=== $e end $(date) rc=$?" >> logs/run_rev.log
done
echo "ALL DONE2 $(date)" >> logs/run_rev.log
