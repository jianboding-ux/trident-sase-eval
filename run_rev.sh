#!/bin/bash
# Revision campaign (resumable).
cd "$(dirname "$0")"
for e in e4c e7 e3c e5b e6b e8; do
  echo "=== $e start $(date)" >> logs/run_rev.log
  python3 trident/run_experiments.py $e >> logs/$e.log 2>&1
  echo "=== $e end $(date) rc=$?" >> logs/run_rev.log
done
echo "ALL DONE $(date)" >> logs/run_rev.log
