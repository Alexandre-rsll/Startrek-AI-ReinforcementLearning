#!/bin/bash
##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-15
## File description:
## repro
##

set -e

mkdir -p runs

echo ""
echo "  LUNAR LANDER - Full launch"

echo ""
echo "[1/7] Random policy (baseline)"
python baseline.py --config configs/baseline_random.yaml

echo ""
echo "[2/7] Heuristic policy (baseline)"
python baseline.py --config configs/baseline_heuristic.yaml

echo ""
echo "[3/7] DQN - training"
python baseline.py --config configs/dqn.yaml

echo ""
echo "[4/7] DQN - eval"
python eval.py --config configs/dqn.yaml

echo ""
echo "[5/7] Double DQN - training"
python baseline.py --config configs/double_dqn.yaml

echo ""
echo "[6/7] Double DQN - eval"
python eval.py --config configs/double_dqn.yaml

echo ""
echo "[7/7] Double DQN PER - multi-seed (5x500+ eval)"
python multi_seed.py --config configs/double_dqn_per.yaml

