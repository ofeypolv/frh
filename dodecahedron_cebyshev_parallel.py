"""
Parallel Chebyshev filtering over all (Ih irrep, parity) combinations.

- Irreps:  Ag, T1g, T2g, Gg, Hg, Au, T1u, T2u, Gu, Hu
- Parity:  even, odd
- Fixed Chebyshev target energy: E_target = -18.0

Each (irrep, parity) is run in a separate process using ProcessPoolExecutor.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# 1. IMPORT YOUR DODECAHEDRON / CHEBYSHEV SETUP
# ---------------------------------------------------------------------------
# This assumes you have a *Python module* dodecahedron_cebyshev.py
# containing your Hamiltonian, helper functions, etc.
#
# If your notebook is called dodecahedron-cebyshev.ipynb, you can:
#   - export it to dodecahedron_cebyshev.py, or
#   - move the relevant definitions into that module.
#
# Adjust the imported names below to match what you actually use.

from dodecahedron_cebyshev import (
    H,                          # Hamiltonian
    N,                          # system size
    projectors,                 # list/array of projectors onto Ih irreps
    irrep_labels,               # corresponding irrep labels (strings)
    make_vc0_parity_Ih_irrep,   # function to build vc0 in given parity & irrep
    check_magnetization_sector, # function to get total magnetization sector
    check_parity_sector,        # function to check parity ("even"/"odd")
    chebyshev_filter_v0_numpy,  # your Chebyshev filter routine
)

# Ug is optional: if it's not in the module, this try/except avoids a crash.
try:
    from dodecahedron_cebyshev import Ug
except ImportError:
    Ug = None

# ---------------------------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------------------------

IRREPS = ["Ag", "T1g", "T2g", "Gg", "Hg",
          "Au", "T1u", "T2u", "Gu", "Hu"]
PARITIES = ["even", "odd"]

# energy window & Chebyshev params
E_MIN = -62.51489576
E_MAX =  62.60812227
E_TARGET = -18.0

CHEB_M = 1000
CHEB_N_STEPS = 50
CHEB_PAD = 0.05
CHEB_USE_JACKSON = True

# If True, each worker saves its final vector to disk as .npy
SAVE_FINAL_VECTORS = True
OUTPUT_DIR = "cheb_results"


# ---------------------------------------------------------------------------
# 3. WORKER FUNCTION: RUN A SINGLE (IRREP, PARITY) JOB
# ---------------------------------------------------------------------------

def run_irrep_parity(irrep, parity,
                     target_E0=E_TARGET,
                     m=CHEB_M,
                     n_steps=CHEB_N_STEPS,
                     pad=CHEB_PAD,
                     use_jackson=CHEB_USE_JACKSON):
    """
    Runs the full pipeline for one (irrep, parity) combination.

    Steps:
        - build vc0 in given parity & irrep
        - diagnostics (magnetization, parity, irrep weights)
        - Chebyshev filtering to target_E0
        - return a small result dict (no huge arrays)

    Uses objects imported from dodecahedron_cebyshev:
        H, N, projectors, irrep_labels,
        make_vc0_parity_Ih_irrep,
        check_magnetization_sector,
        check_parity_sector,
        chebyshev_filter_v0_numpy,
        (optionally Ug)
    """

    print(f"\n=== Starting job: irrep={irrep}, parity={parity} ===")

    # 1) Build initial state in this irrep & parity
    vc0 = make_vc0_parity_Ih_irrep(
        N,
        projectors,
        par=parity,
        irrep_labels=irrep_labels,
        target_irrep=irrep,
        rng=None
    )

    # quick checks
    norm_v = np.vdot(vc0, vc0).real
    mag = check_magnetization_sector(vc0, N)
    pari = check_parity_sector(vc0, N)
    support0 = int(np.count_nonzero(np.abs(vc0) > 1e-10))

    print(f"[{irrep}, {parity}] norm = {norm_v:.6e}")
    print(f"[{irrep}, {parity}] magnetization (expected 0): {mag}")
    print(f"[{irrep}, {parity}] parity sector (expected {parity!r}): {pari}")
    print(f"[{irrep}, {parity}] initial support size: {support0}")

    # irrep weights
    weights = {}
    for P, lbl in zip(projectors, irrep_labels):
        psi = P @ vc0
        weights[lbl] = float(np.vdot(psi, psi).real)

    sorted_weights = sorted(weights.items(), key=lambda kv: -kv[1])
    print(f"\n[{irrep}, {parity}] Irrep weights (descending):")
    for lbl, w in sorted_weights:
        print(f"  {lbl:>4}: {w:.6e}")

    total_weight = sum(weights.values())
    print(f"[{irrep}, {parity}] Sum of irrep-weights = {total_weight:.12f} (≈1.0)")

    if abs(total_weight - 1.0) > 1e-6:
        print(f"[{irrep}, {parity}] WARNING: sum of irrep weights deviates from 1 by {total_weight - 1.0}")

    # optional Ug diagnostic if Ug is available
    if Ug is not None:
        try:
            resid = np.linalg.norm(Ug @ vc0 - vc0)
            ov = (np.vdot(vc0, Ug @ vc0) / np.vdot(vc0, vc0)).real
            print(f"[{irrep}, {parity}] Ug diag: ||Ug v - v|| = {resid:.2e}, <v|Ug|v> = {ov:.6f}")
        except Exception as e:
            print(f"[{irrep}, {parity}] Ug diagnostic failed: {e}")

    # 2) Chebyshev filtering
    Emin = E_MIN
    Emax = E_MAX

    current = vc0
    last_eval = None

    print(f"[{irrep}, {parity}] Chebyshev: n_steps={n_steps}, m={m}, pad={pad}, target_E0={target_E0}")

    try:
        for i in range(1, n_steps + 1):
            Phi, evals = chebyshev_filter_v0_numpy(
                H, current,
                Emin=Emin,
                Emax=Emax,
                target_E0=target_E0,
                m=m,
                pad=pad,
                use_jackson=use_jackson
            )

            if not np.isfinite(evals):
                print(f"[{irrep}, {parity}] stopped at step {i}: non-finite eval {evals}")
                break
            if not np.all(np.isfinite(Phi)):
                print(f"[{irrep}, {parity}] stopped at step {i}: non-finite entries in Phi")
                break

            current = Phi
            last_eval = evals

            if (i % max(1, n_steps // 10)) == 0 or i == 1 or i == n_steps:
                sup = int(np.count_nonzero(np.abs(current) > 1e-10))
                print(f"[{irrep}, {parity}] step {i:4d}: approx E = {last_eval:.8f}, support = {sup}")

    except RuntimeError as e:
        print(f"[{irrep}, {parity}] RuntimeError during filtering: {e}")

    E_final = float(last_eval) if last_eval is not None else None
    final_support = int(np.count_nonzero(np.abs(current) > 1e-10))

    print(f"[{irrep}, {parity}] Done. Final approx E: {E_final}")
    print(f"[{irrep}, {parity}] Final support size: {final_support}")

    # optionally save final vector to disk (avoid returning giant arrays)
    if SAVE_FINAL_VECTORS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fname = os.path.join(OUTPUT_DIR, f"Phi_final_{irrep}_{parity}.npy")
        np.save(fname, current)
        print(f"[{irrep}, {parity}] Saved final vector to {fname}")

    # Return only small metadata
    return {
        "irrep": irrep,
        "parity": parity,
        "norm_v": norm_v,
        "mag": mag,
        "parity_detected": pari,
        "initial_support": support0,
        "final_support": final_support,
        "E_final": E_final,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# 4. MAIN: PARALLEL EXECUTION OVER ALL (IRREP, PARITY)
# ---------------------------------------------------------------------------

def main():
    # all (irrep, parity) combinations
    jobs = [(ir, par) for ir in IRREPS for par in PARITIES]

    # choose number of workers (one per core or up to #jobs)
    num_workers = min(len(jobs), os.cpu_count() or 1)
    print(f"Launching {num_workers} parallel workers for {len(jobs)} jobs")

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_job = {
            executor.submit(run_irrep_parity, ir, par): (ir, par)
            for (ir, par) in jobs
        }

        for future in as_completed(future_to_job):
            ir, par = future_to_job[future]
            try:
                res = future.result()
                results.append(res)
                print(f"[{ir}, {par}] COMPLETED: E_final={res['E_final']}")
            except Exception as e:
                print(f"[{ir}, {par}] FAILED with error: {e}")

    # Summary
    print("\n================ SUMMARY OVER ALL JOBS ================")
    for r in sorted(results, key=lambda x: (x["parity"], x["irrep"])):
        print(
            f"{r['irrep']:>3}  {r['parity']:>4} | "
            f"E_final = {r['E_final']}, "
            f"support = {r['final_support']}"
        )


if __name__ == "__main__":
    # Optional: limit BLAS threads to avoid oversubscription if you see slowdown
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")

    main()