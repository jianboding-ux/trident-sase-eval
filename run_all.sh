#!/bin/bash
# Runs the complete TRIDENT experiment campaign sequentially (single-node reference deployment). Resumable.
cd "$(dirname "$0")"
for e in e1 e3 e3p e4 e4a e5 e6 e1b; do
  echo "=== $e start $(date)" >> logs/run_all.log
  python3 trident/run_experiments.py $e >> logs/$e.log 2>&1
  echo "=== $e end $(date) rc=$?" >> logs/run_all.log
done
echo "ALL DONE $(date)" >> logs/run_all.log
