#!/bin/sh
# Reproduce the project's five results (Table 1, Fig. 1, Fig. 2, Table 2, Table 3) and the paired tests.
#
#   sh scripts/reproduce.sh            rebuild tables, figures and tests from the committed CSVs (~1 min)
#   sh scripts/reproduce.sh --full     re-run every training run first, then rebuild (days on one laptop)
#   WORKERS=4 sh scripts/reproduce.sh --full     parallel runs (each needs ~400 MB RAM and one CPU core)
#
# Data needs no download: every run simulates its trajectories with SciPy (src/pinn_swing/sim.py).
# Full runs are restart-safe: re-running skips runs whose JSON already exists in results/raw/.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
WORKERS=${WORKERS:-1}

if [ "$1" = "--full" ]; then
    $PY -m pytest -q
    # Table 1, forward part: 3 seeds each, the authors' settings (configs/forward.yaml)
    mkdir -p results/replication
    for lam in 1 0; do
        for seed in 0 1 2; do
            out=results/replication/forward_lam${lam}_seed${seed}.json
            [ -f "$out" ] || $PY scripts/run_forward.py --lam $lam --seed $seed --out "$out"
        done
    done
    # the six sweeps (960 runs); --min-free-mem waits whenever free RAM drops below 25%
    for s in S1_size_noise_boundary S2_size_noise_interior S3_physics_weight S4_collocation \
             S5_extrapolation S6_identification; do
        $PY scripts/run_sweep.py configs/experiments/$s.yaml --workers "$WORKERS" --min-free-mem 25
        $PY scripts/aggregate.py $s
    done
fi

$PY scripts/summarize_replication.py > /dev/null   # Table 1
$PY scripts/make_figures.py                        # Fig. 1, Fig. 2, Table 2, Table 3
$PY scripts/stats.py > /dev/null                   # paired Wilcoxon tests
echo "done: results/table1_replication.md, figures/fig1_*.png, figures/fig2_*.png,"
echo "      results/table2_identification.md, results/table3_design_choices.md, results/stats_paired_tests.md"
