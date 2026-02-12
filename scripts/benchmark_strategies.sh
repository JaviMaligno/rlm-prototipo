#!/usr/bin/env bash
# Benchmark: Global budget vs Refill-per-turn
# Runs the same question with both strategies and compares results.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1

QUESTION="Resume los 3 temas principales de los papers, con una explicacion de 2-3 frases por tema"
INPUT="data/*.txt"
MAX_TURNS=5
MAX_SUBCALLS=15

echo "=========================================="
echo "  STRATEGY A: Global budget (current)"
echo "  max_turns=$MAX_TURNS, max_subcalls=$MAX_SUBCALLS"
echo "=========================================="
time rlm run --input "$INPUT" --question "$QUESTION" \
    --max-turns $MAX_TURNS --max-subcalls $MAX_SUBCALLS \
    2>&1 | tee /tmp/rlm_strategy_a.log

echo ""
echo "=========================================="
echo "  STRATEGY B: Refill per turn"
echo "  max_turns=$MAX_TURNS, max_subcalls=$MAX_SUBCALLS (per turn)"
echo "=========================================="
time rlm run --input "$INPUT" --question "$QUESTION" \
    --max-turns $MAX_TURNS --max-subcalls $MAX_SUBCALLS \
    --refill-per-turn \
    2>&1 | tee /tmp/rlm_strategy_b.log

echo ""
echo "=========================================="
echo "  COMPARISON SUMMARY"
echo "=========================================="
echo "Strategy A (global budget):"
grep -E "(Completed in|Final Answer|Max turns reached)" /tmp/rlm_strategy_a.log | tail -3
echo ""
echo "Strategy B (refill per turn):"
grep -E "(Completed in|Final Answer|Max turns reached)" /tmp/rlm_strategy_b.log | tail -3
