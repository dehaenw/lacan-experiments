"""
run_guacamol_distribution.py — Distribution learning benchmark runner for LACAN.

Usage
-----
    python run_guacamol_distribution.py \\
        --training_file data/guacamol_v1_train.smiles \\
        --output distribution_results.json \\
        --n_samples 10000 \\
        --jobs -1

The distribution learning benchmark tests five properties of the generated
molecule set against the ChEMBL training distribution:
  - KL divergence of physicochemical descriptors
  - Frechet ChemNet Distance (FCD)
  - Nearest-neighbour similarity
  - Fragment similarity (Bemis-Murcko)
  - Scaffold similarity (Bemis-Murcko)

Dependency note — FCD version
------------------------------
GuacaMol's FCD benchmark requires fcd==1.1 (the old Keras/h5 model format).
FCD >=1.2 switched to PyTorch and uses a different model file format, causing:

    UnpicklingError: invalid load key, 'H'

If you see this error, check your installed version:

    pip show fcd

And force downgrade if needed:

    pip install "fcd==1.1" --force-reinstall

This script detects the FCD version at startup and warns loudly if it looks
wrong, before wasting time generating molecules.
"""

import argparse
import json
import logging
import sys
import time
from collections import OrderedDict

from rdkit import Chem

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("lacan_guacamol_dist")


# ── FCD version check ─────────────────────────────────────────────────────────

def _check_fcd_version():
    """Warn clearly if fcd version is incompatible with GuacaMol's FCD benchmark.

    GuacaMol requires fcd==1.1. fcd>=1.2 uses PyTorch model format which
    causes UnpicklingError when GuacaMol tries to load the Keras h5 model.
    """
    try:
        import fcd
        version = getattr(fcd, "__version__", None)
        if version is None:
            # Try importlib as fallback
            from importlib.metadata import version as get_version
            version = get_version("fcd")
        major, minor = int(version.split(".")[0]), int(version.split(".")[1])
        if major > 1 or (major == 1 and minor > 1):
            logger.warning(
                f"fcd version {version} detected. GuacaMol's FCD benchmark "
                f"requires fcd==1.1. You may see 'UnpicklingError: invalid "
                f"load key' when the FCD benchmark runs. Fix with:\n"
                f"    pip install 'fcd==1.1' --force-reinstall"
            )
            return False
        else:
            logger.info(f"fcd version {version} — OK for GuacaMol.")
            return True
    except Exception as e:
        logger.warning(f"Could not determine fcd version: {e}. Proceeding anyway.")
        return True


# ── imports ───────────────────────────────────────────────────────────────────

from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.assess_distribution_learning import assess_distribution_learning

from lacan import gen, lacan as lacan_module

PROFILE = lacan_module.load_profile("chembl_full")


# ── Distribution generator ────────────────────────────────────────────────────

class LacanDistributionGenerator(DistributionMatchingGenerator):
    """Wraps LACAN's random generator as a GuacaMol DistributionMatchingGenerator.

    Generates a large cache of filtered molecules upfront, then serves slices
    on each call to generate(). This avoids re-running the (parallel) generation
    on every benchmark call, since GuacaMol calls generate() multiple times
    with different sample sizes during the benchmark suite.

    Parameters
    ----------
    profile          : LACAN profile dict
    fragcorpus       : fragment corpus (default: ChEMBL rls.csv)
    n_jobs           : parallel workers (-1 = all cores)
    seed             : random seed
    cache_multiplier : generate this multiple of the largest expected
                       number_samples upfront (default 3). GuacaMol's
                       distribution benchmarks typically request 10000 samples;
                       cache_multiplier=3 pre-generates 30000 molecules so
                       repeated calls get fresh non-overlapping slices.
    threshold        : minimum LACAN score for generated molecules (default 0.001
                       — very permissive, keeps generation fast while filtering
                       clearly invalid structures)
    min_atoms        : minimum heavy atom count (default 14)
    max_atoms        : maximum heavy atom count (default 45)
    """

    def __init__(self, profile, fragcorpus=None, n_jobs=-1, seed=42,
                 cache_multiplier=3, threshold=0.001, min_atoms=14, max_atoms=45):
        self.profile = profile
        self.fragcorpus = fragcorpus
        self.n_jobs = n_jobs
        self.seed = seed
        self.cache_multiplier = cache_multiplier
        self.threshold = threshold
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self._cache = None
        self._call_count = 0

    def generate(self, number_samples: int):
        """Return a list of SMILES strings of length number_samples.

        Generates a cache of cache_multiplier * number_samples molecules on
        the first call, then serves rotating slices on subsequent calls.
        """
        total_needed = number_samples * self.cache_multiplier

        if self._cache is None or len(self._cache) < number_samples:
            logger.info(
                f"Generating cache of {total_needed} molecules "
                f"(threshold={self.threshold}, n_jobs={self.n_jobs})..."
            )
            t0 = time.time()
            mols = gen.generate_filtered_molecules(
                self.profile,
                fragcorpus=self.fragcorpus,
                threshold=self.threshold,
                min_atoms=self.min_atoms,
                max_atoms=self.max_atoms,
                n_molecules=total_needed,
                n_jobs=self.n_jobs,
                seed=self.seed,
            )
            self._cache = [Chem.MolToSmiles(m) for m in mols if m is not None]
            logger.info(
                f"Generated and cached {len(self._cache)} molecules "
                f"in {time.time()-t0:.1f}s"
            )

        # Serve a rotating slice so repeated calls get different molecules
        n = len(self._cache)
        start = (self._call_count * number_samples) % n
        end = start + number_samples
        self._call_count += 1

        if end <= n:
            return self._cache[start:end]
        else:
            # Wrap around
            return (self._cache[start:] + self._cache[:end - n])[:number_samples]


# ── benchmark runner ──────────────────────────────────────────────────────────

def run_distribution_benchmark(
        training_file,
        output_file="distribution_results.json",
        n_jobs=-1,
        seed=42,
        cache_multiplier=3,
        threshold=0.001,
        min_atoms=14,
        max_atoms=45,
):
    """Run the GuacaMol distribution learning benchmark suite.

    Parameters
    ----------
    training_file    : str — path to guacamol_v1_train.smiles
    output_file      : str — path to write JSON results
    n_jobs           : int — parallel workers for LACAN generation
    seed             : int — random seed
    cache_multiplier : int — pre-generate this multiple of requested samples
    threshold        : float — minimum LACAN score filter
    min_atoms        : int — minimum heavy atom count
    max_atoms        : int — maximum heavy atom count

    Returns
    -------
    benchmark result object (from GuacaMol)
    """
    logger.info("Checking FCD version compatibility...")
    _check_fcd_version()

    logger.info(
        f"Starting GuacaMol distribution learning benchmark\n"
        f"  training_file={training_file}\n"
        f"  output_file={output_file}\n"
        f"  n_jobs={n_jobs}  seed={seed}  threshold={threshold}\n"
        f"  cache_multiplier={cache_multiplier}"
    )

    generator = LacanDistributionGenerator(
        profile=PROFILE,
        n_jobs=n_jobs,
        seed=seed,
        cache_multiplier=cache_multiplier,
        threshold=threshold,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
    )

    t0 = time.time()
    try:
        result = assess_distribution_learning(
            generator,
            chembl_training_file=training_file,
            json_output_file=output_file,
        )
    except Exception as e:
        # Provide a helpful message for the common FCD version error
        msg = str(e)
        if "invalid load key" in msg or "UnpicklingError" in msg:
            logger.error(
                f"FCD model loading failed — this is almost certainly a version "
                f"mismatch. GuacaMol requires fcd==1.1.\n"
                f"Fix: pip install 'fcd==1.1' --force-reinstall\n"
                f"Original error: {e}"
            )
        else:
            logger.error(f"Benchmark failed: {e}")
        raise

    elapsed = time.time() - t0
    logger.info(f"Distribution benchmark completed in {elapsed/60:.1f} min")
    logger.info(f"Results saved to {output_file}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run GuacaMol distribution learning benchmark with LACAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_file",
        required=True,
        help="Path to guacamol_v1_train.smiles",
    )
    parser.add_argument(
        "--output", default="distribution_results.json",
        help="Path for JSON output",
    )
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel workers for LACAN generation (-1 = all cores)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--cache_multiplier", type=int, default=3,
        help="Pre-generate this multiple of the requested sample count",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.001,
        help="Minimum LACAN score for generated molecules",
    )
    parser.add_argument(
        "--min_atoms", type=int, default=14,
        help="Minimum heavy atom count",
    )
    parser.add_argument(
        "--max_atoms", type=int, default=45,
        help="Maximum heavy atom count",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_distribution_benchmark(
        training_file=args.training_file,
        output_file=args.output,
        n_jobs=args.jobs,
        seed=args.seed,
        cache_multiplier=args.cache_multiplier,
        threshold=args.threshold,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
    )

