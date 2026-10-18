"""Run every (setting, seed) of one experiment config in parallel. Restart-safe: finished runs are skipped.

Examples (from the repo root):
  python scripts/run_sweep.py configs/experiments/S1_size_noise_boundary.yaml --dry-run
  python scripts/run_sweep.py configs/experiments/S1_size_noise_boundary.yaml --workers 8
  python scripts/run_sweep.py configs/experiments/S6_identification.yaml --shard 0/4   # 1 of 4 laptops

Each run writes results/raw/<name>/<run_id>.json; failures are logged next to it as <run_id>.log.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {"forward": ROOT / "scripts" / "run_forward.py",
           "identification": ROOT / "scripts" / "run_identification.py"}


def expand(exp):
    """List of (run_id, args dict) for an experiment config: fixed + (grid product or variants) x seeds."""
    grid = exp.get("grid", {})
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())] if grid else [{}]
    variants = exp.get("variants", [{}])
    runs = []
    for combo, var, seed in itertools.product(combos, variants, exp["seeds"]):
        args = {**exp.get("fixed", {}), **combo, **var, "seed": seed}
        varying = {**combo, **var, "seed": seed}
        run_id = "_".join(f"{k}{v}" for k, v in varying.items())
        runs.append((run_id, args))
    return runs


def expected_cost(args):
    """Rough ordering key: start the long physics runs first so the pool finishes evenly."""
    if args.get("method") == "two_stage":
        return 1
    return 3 if (args.get("lam", 1) or 0) > 0 else 0


def command(script, args, out, overrides):
    cmd = [sys.executable, str(SCRIPTS[script]), "--threads", "1", "--out", str(out)]
    for k, v in {**args, **overrides}.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    p.add_argument("--shard", default="0/1", help="i/n: run only every n-th run starting at i (split across machines)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override an argument for every run, e.g. max_iter=50 for a smoke test")
    p.add_argument("--out-root", default=str(ROOT / "results" / "raw"))
    a = p.parse_args(argv)

    exp = yaml.safe_load(open(a.config))
    overrides = dict(kv.split("=", 1) for kv in a.set)
    out_dir = Path(a.out_root) / exp["name"]
    i, n = map(int, a.shard.split("/"))
    runs = expand(exp)[i::n]
    todo = [(rid, args) for rid, args in runs if not (out_dir / f"{rid}.json").exists()]
    todo.sort(key=lambda r: -expected_cost(r[1]))
    print(f"{exp['name']}: {len(runs)} runs in this shard, {len(runs) - len(todo)} already done, {len(todo)} to run")
    if a.dry_run:
        for rid, args in todo:
            print(" ", rid, args)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(rid, args):
        out = out_dir / f"{rid}.json"
        proc = subprocess.run(command(exp["script"], args, out, overrides), capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            (out_dir / f"{rid}.log").write_text(proc.stdout + proc.stderr)
        return rid, proc.returncode

    start, done, errors = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(one, rid, args) for rid, args in todo]
        for fut in as_completed(futures):
            rid, code = fut.result()
            done += 1
            errors += code != 0
            elapsed = time.time() - start
            eta = elapsed / done * (len(todo) - done)
            print(f"[{done}/{len(todo)}] {'ERROR ' if code else ''}{rid}  elapsed {elapsed / 60:.1f} min, "
                  f"eta {eta / 60:.1f} min", flush=True)
    print(f"finished: {done - errors} ok, {errors} errors")


if __name__ == "__main__":
    main()
