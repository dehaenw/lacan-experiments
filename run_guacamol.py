"""
run_guacamol.py — Goal-directed GuacaMol benchmark runner for LACAN.

Usage
-----
    python run_guacamol.py [--output results.json] [--version v2] [--jobs -1]

The script uses the ``guacamol`` preset (popsize=200, mutation-dominant
budget allocation, inverted plateau logic) and reports per-benchmark scores
plus the mean score at the end.

All scoring is blind: the generator receives only a callable that returns a
float — it has no access to benchmark names or task metadata.
"""

import argparse
import os
import json
import logging
import time
from collections import OrderedDict

from rdkit import Chem

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("lacan_guacamol")

# ── imports ───────────────────────────────────────────────────────────────────
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation

from lacan import gen, lacan as lacan_module


# ── profile ───────────────────────────────────────────────────────────────────
# Standard drug-likeness profile.  Adjust as needed for your installation.
PROFILE = lacan_module.load_profile("chembl_full")


# ── GuacaMol adapter ──────────────────────────────────────────────────────────

class LacanGoalDirectedGenerator(GoalDirectedGenerator):
    """Wraps LACAN's GA as a GuacaMol GoalDirectedGenerator.

    The scoring function received from GuacaMol is forwarded directly to
    ``gen.generate_optimized_molecules`` as-is.  No benchmark-specific
    knowledge leaks into the generator: it only sees a callable that maps
    a list of RDKit Mol objects to a list of floats.

    Cross-task warm-starting: the top molecules from each completed task's
    HoF are re-scored on the next task and used to seed its initial pool.
    This avoids starting from scratch every task. The molecules are re-scored
    (not assumed to be good) so there is no information leakage — a molecule
    that scored well on Celecoxib rediscovery may score poorly on Sitagliptin
    MPO and will simply be outcompeted in the initial pool cull.

    Parameters
    ----------
    profile        : LACAN profile dict
    preset         : str — GA preset (default ``"guacamol"``)
    n_jobs         : int — parallel workers (-1 = all cores)
    seed_pool_size : int — how many HoF molecules to carry over between tasks
    reporter       : bool — whether to attach a GAReporter for post-run plots
    extra_kwargs   : forwarded verbatim to generate_optimized_molecules
    """

    def __init__(self, profile, preset="guacamol", n_jobs=-1,
                 cache_dir=None, reporter=False, **extra_kwargs):
        self.profile = profile
        self.preset = preset
        self.n_jobs = n_jobs
        self.cache_dir = cache_dir
        self.attach_reporter = reporter
        self.extra_kwargs = extra_kwargs
        self.reporters = []
        self._current_benchmark_name = None

    def generate_optimized_molecules(
            self,
            scoring_function,
            number_molecules: int,
            starting_population=None,
    ):
        def _score_fn(mols):
            scores = []
            for mol in mols:
                try:
                    smi = Chem.MolToSmiles(mol)
                    s = scoring_function.score(smi)
                    scores.append(float(s) if s is not None else 0.0)
                except Exception:
                    scores.append(0.0)
            return scores

        # Load starting population from per-task cache JSONs.
        # GuacaMol passes starting_population=None, so we load from cache_dir ourselves.
        # Files named e.g. guacamol_cache/Scaffold_Hop.json with {"top": [{"smiles": ...}]}
        cache_smiles = list(starting_population) if starting_population else []

        if not cache_smiles and self.cache_dir and self._current_benchmark_name:
            cache_path = os.path.join(self.cache_dir, self._current_benchmark_name + ".json")
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    data = json.load(f)
                cache_smiles = [e["smiles"] for e in data.get("top", []) if "smiles" in e]
                logger.info(f"  Loaded {len(cache_smiles)} molecules from {cache_path}")
            else:
                logger.warning(f"  No cache file at {cache_path}")
        seed_mols = None
        if cache_smiles:
            seed_mols = [m for smi in cache_smiles
                         for m in [Chem.MolFromSmiles(smi)] if m is not None]
            seed_mols = seed_mols or None
            logger.info(f"  Seeding GA with {len(seed_mols) if seed_mols else 0} molecules")

        reporter = None
        if self.attach_reporter:
            reporter = gen.GAReporter(label=f"benchmark_{len(self.reporters)+1}")
            self.reporters.append(reporter)

        results = gen.generate_optimized_molecules(
            scoring_function=_score_fn,
            profile=self.profile,
            preset=self.preset,
            n_jobs=self.n_jobs,
            seed_mols=seed_mols,
            callback=reporter,
            **self.extra_kwargs,
        )

        top_smiles = [smi for smi, _ in results[:number_molecules]]

        if len(top_smiles) < number_molecules:
            logger.warning(
                f"Only {len(top_smiles)} molecules returned; "
                f"padding to {number_molecules} with random molecules."
            )
            try:
                extras = gen.generate_filtered_molecules(
                    self.profile,
                    n_molecules=number_molecules - len(top_smiles),
                    n_jobs=self.n_jobs,
                )
                for m in extras:
                    try:
                        top_smiles.append(Chem.MolToSmiles(m))
                    except Exception:
                        pass
            except Exception:
                pass

        return top_smiles[:number_molecules]


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(output_file="results.json", version="v2", n_jobs=-1,
                  cache_dir=None, reporter=False, **ga_kwargs):
    """Run the full GuacaMol goal-directed benchmark suite.

    Parameters
    ----------
    output_file : str — path to write JSON results
    version     : str — benchmark suite version (default "v2")
    n_jobs      : int — parallel workers for LACAN (-1 = all cores)
    reporter    : bool — attach GAReporter to each benchmark run
    **ga_kwargs : forwarded to generate_optimized_molecules (override preset)

    Returns
    -------
    list of BenchmarkResult
    """
    logger.info(f"Starting GuacaMol goal-directed benchmark (version={version})")
    logger.info(f"GA kwargs: preset=guacamol, n_jobs={n_jobs}, overrides={ga_kwargs}")

    generator = LacanGoalDirectedGenerator(
        profile=PROFILE,
        preset="guacamol",
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        reporter=reporter,
        **ga_kwargs,
    )

    from guacamol.benchmark_suites import goal_directed_benchmark_suite

    benchmarks = goal_directed_benchmark_suite(version_name=version)
    results = []
    scores = []

    for i, benchmark in enumerate(benchmarks, 1):
        logger.info(f"Running benchmark {i}/{len(benchmarks)}: {benchmark.name}")
        generator._current_benchmark_name = benchmark.name
        t0 = time.time()
        result = benchmark.assess_model(generator)
        elapsed = time.time() - t0
        logger.info(f"  Score: {result.score:.4f}  |  Time: {elapsed/60:.1f} min")
        results.append(result)
        scores.append(result.score)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"Mean score across {len(scores)} benchmarks: {mean_score:.4f}")
    logger.info("Per-benchmark scores:")
    for r, s in zip(results, scores):
        logger.info(f"  {r.benchmark_name:<45s}  {s:.4f}")

    benchmark_results = {
        "guacamol_version": version,
        "mean_score": mean_score,
        "results": [r.__dict__ for r in results],
    }
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_file}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run GuacaMol goal-directed benchmark with LACAN GA."
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Path for JSON output (default: results.json)"
    )
    parser.add_argument(
        "--version", default="v2",
        help="GuacaMol benchmark suite version (default: v2)"
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers for LACAN (-1 = all cores, default: -1)"
    )
    parser.add_argument(
        "--cache_dir", default="guacamol_cache",
        help="Directory with per-task cache JSONs (default: guacamol_cache)"
    )
    parser.add_argument(
        "--reporter", action="store_true",
        help="Attach GAReporter to each benchmark run for diagnostics"
    )
    # GA parameter overrides
    parser.add_argument("--popsize",        type=int,   default=None)
    parser.add_argument("--startN",         type=int,   default=None)
    parser.add_argument("--generations",    type=int,   default=None)
    parser.add_argument("--scoring_budget", type=int,   default=None)
    parser.add_argument("--explore_fraction", type=float, default=None,
                        dest="base_explore_fraction")
    parser.add_argument("--plateau_patience", type=int, default=None)
    parser.add_argument("--hof_size",       type=int,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Collect only the overrides that were actually passed
    ga_overrides = {k: v for k, v in vars(args).items()
                    if k not in ("output", "version", "jobs", "reporter", "cache_dir")
                    and v is not None}

    run_benchmark(
        output_file=args.output,
        version=args.version,
        n_jobs=args.jobs,
        cache_dir=args.cache_dir,
        reporter=args.reporter,
        **ga_overrides,
    )
