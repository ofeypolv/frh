#!/usr/bin/env python3
# dodecahedron_chebyshev_solver.py
#
# This file does ALL of the following, using your conventions:
#
# 1. Defines:
#    - Pauli X, Pauli Z
#    - dodecahedral_bonds() with your 20-site connectivity
#
# 2. Builds THREE physics pieces in the exact style you asked for:
#    (a) ising_xx_dodecahedron(N, J)
#          H_int = J * sum_{(i,j) in bonds} σ_i^x σ_j^x
#    (b) transverse_field_z_dodecahedron(N, h)
#          H_field = -h * sum_i σ_i^z
#    (c) full_tfim_dodecahedron(N, J, h)
#          H = H_int + H_field
#
#    These versions use explicit kron with SciPy and literally build the
#    (2^N x 2^N) matrix. For N=20 this is not computationally realistic,
#    but they exactly match your math and are good for small-N tests.
#
# 3. Builds a scalable PETSc shell matrix H_shell for the SAME Hamiltonian:
#
#        H = J * Σ_{(i,j)} σ_i^x σ_j^x  -  h * Σ_i σ_i^z
#
#    in the computational Z basis (bitstrings length N). This does NOT build
#    the giant matrix. Instead, it only defines MatShell.mult(), i.e. how to
#    apply H to a vector. This is what we actually send to the Chebyshev solver.
#
# 4. Implements a Chebyshev-filtered subspace iteration (Chebyshev polynomial
#    filtering + Rayleigh-Ritz) with slepc4py/petsc4py to get eigenvalues
#    closest to a target interior energy (e.g. μ = -18) inside a known
#    spectral window (e.g. [-62, 62]).
#
# HOW TO RUN (single MPI rank first!):
#   mpirun -n 1 python dodecahedron_chebyshev_solver.py
#
# You must set the parameters in main():
#   N_SPINS, COUPLING_J, FIELD_h, TARGET_MU, etc.
#
# IMPORTANT LIMITATION:
# - orthonormalize_block() here is only safe on 1 MPI rank (COMM_WORLD size = 1).
#   Run with mpirun -n 1 for now. Parallel QR would be the next step.
#
# Author: GPT-5 Thinking
#

import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from slepc4py import SLEPc  # imported for completeness if you later want to use SLEPc solvers directly
from petsc4py import PETSc


# -----------------------------------------------------------------------------
# 1. Geometry and Pauli helpers
# -----------------------------------------------------------------------------

def pauli_x():
    """Pauli X matrix."""
    return np.array([[0, 1], [1, 0]], dtype=np.float64)

def pauli_z():
    """Pauli Z matrix."""
    return np.array([[1, 0], [0, -1]], dtype=np.float64)

def dodecahedral_bonds():
    """
    Connectivity (edges) of the 20-vertex dodecahedral graph.
    Sites are 0-based indices in [0,19].

    Returns:
        list[(int,int)]
    """
    bonds = [
        (0, 13), (0, 14), (0, 15),
        (1, 4), (1, 5), (1, 12),
        (2, 6), (2, 13), (2, 18),
        (3, 7), (3, 14), (3, 19),
        (4, 10), (4, 18),
        (5, 11), (5, 19),
        (6, 10), (6, 15),
        (7, 11), (7, 15),
        (8, 9), (8, 13), (8, 16),
        (9, 14), (9, 17),
        (10, 11),
        (12, 16), (12, 17),
        (16, 18),
        (17, 19)
    ]
    return bonds


# -----------------------------------------------------------------------------
# 2. Direct sparse matrix builders (your literal kron approach)
#    WARNING: these explode in size for N=20, but we keep them for clarity/testing.
# -----------------------------------------------------------------------------

def ising_xx_dodecahedron(N, J):
    """
    Interaction term ONLY:

        H_int = J * sum_{(i,j) in bonds} σ_i^x σ_j^x

    Where σ_i^x acts as Pauli X on site i (and identity elsewhere).

    Parameters:
        N (int): number of spins (must be 20 for the dodecahedron)
        J (float): coupling strength

    Returns:
        H_int (csr_matrix): shape (2**N, 2**N)
    """
    if N != 20:
        raise ValueError("This model assumes N = 20 sites (the dodecahedron).")

    I = identity(2, format="csr", dtype=np.float64)
    X = csr_matrix(pauli_x(), dtype=np.float64)

    H_int = csr_matrix((2**N, 2**N), dtype=np.float64)
    bonds = dodecahedral_bonds()

    for (i, j) in bonds:
        term = 1
        for site in range(N):
            if site == i or site == j:
                term = kron(term, X, format="csr")
            else:
                term = kron(term, I, format="csr")
        H_int = H_int + J * term

    return H_int


def transverse_field_z_dodecahedron(N, h):
    """
    Field term ONLY:

        H_field = -h * sum_i σ_i^z

    Where σ_i^z acts as Pauli Z on site i (and identity elsewhere).

    Parameters:
        N (int): number of spins (must be 20)
        h (float): field strength

    Returns:
        H_field (csr_matrix): shape (2**N, 2**N)
    """
    if N != 20:
        raise ValueError("This model assumes N = 20 sites (the dodecahedron).")

    I = identity(2, format="csr", dtype=np.float64)
    Z = csr_matrix(pauli_z(), dtype=np.float64)

    H_field = csr_matrix((2**N, 2**N), dtype=np.float64)

    for i in range(N):
        term = 1
        for site in range(N):
            if site == i:
                term = kron(term, Z, format="csr")
            else:
                term = kron(term, I, format="csr")
        H_field = H_field + (-h) * term

    return H_field


def full_tfim_dodecahedron(N, J, h):
    """
    Combined Hamiltonian in your convention:

        H =  J * sum_{(i,j) in bonds} σ_i^x σ_j^x
           - h * sum_i σ_i^z

    Parameters:
        N (int): number of spins (must be 20)
        J (float)
        h (float)

    Returns:
        H (csr_matrix): shape (2**N, 2**N)
    """
    H_int = ising_xx_dodecahedron(N, J)
    H_field = transverse_field_z_dodecahedron(N, h)
    return H_int + H_field


# -----------------------------------------------------------------------------
# 3. PETSc shell matrix version of the SAME Hamiltonian
#
#    We DON'T explicitly build the giant matrix for N=20.
#    Instead we define how H acts on a vector in the computational Z basis.
#
#    Basis: bitstrings b in [0, 2^N - 1].
#       - σ_i^z |b> = (+1 if bit i == 0 else -1) |b>
#       - σ_i^x σ_j^x |b> = |b with bits i and j flipped|
#
#    Hamiltonian:
#
#        H =  J * Σ_{(i,j)∈bonds} X_i X_j
#           -  h * Σ_i Z_i
#
#    This matches full_tfim_dodecahedron() physically.
# -----------------------------------------------------------------------------

class DodecahedronHamiltonianShellCtx:
    def __init__(self, N, J, h, comm):
        """
        N : number of spins (expected 20)
        J : coupling for Σ X_i X_j
        h : field for Σ Z_i (note the Hamiltonian uses -h Σ Z_i)
        comm : PETSc communicator
        """
        self.N = N
        self.J = J
        self.h = h
        self.comm = comm
        self.bonds = dodecahedral_bonds()

        # Hilbert space dimension = 2^N
        self.dim = 1 << N  # same as 2**N

    def mult(self, A, x, y):
        """
        y = H x
        where H = J Σ_{(i,j)} X_i X_j - h Σ_i Z_i.

        x, y are PETSc.Vec with global size 2^N.
        This routine:
          - zeroes y
          - loops over owned basis states b
          - accumulates all contributions to y
          - does vector assembly
        """

        N = self.N
        J = self.J
        h = self.h
        bonds = self.bonds

        # start from zero
        y.set(0.0)

        # local ownership range for this rank
        istart, iend = x.getOwnershipRange()  # global indices handled locally

        # read local slice of x
        x_local = x.getArray(readonly=True)

        # We'll stage contributions in Python lists, then dump them into y with add mode.
        y_idx = []
        y_val = []

        for b in range(istart, iend):
            loc = b - istart
            amp_b = x_local[loc]

            # ----- diagonal Z part -----
            # Z_i|b> = (+1 if bit_i==0, -1 if bit_i==1) |b>.
            # Contribution: (-h) * sum_i Z_i(b) * amp_b on |b>.
            z_sum = 0.0
            for i in range(N):
                bit_i = (b >> i) & 1
                z_sum += (1.0 if bit_i == 0 else -1.0)
            diag_contrib = (-h) * z_sum * amp_b
            if diag_contrib != 0.0:
                y_idx.append(b)
                y_val.append(diag_contrib)

            # ----- off-diagonal XX part -----
            # For each bond (i,j), X_i X_j flips bits i and j.
            # That sends amplitude amp_b to |b_flip>.
            for (i, j) in bonds:
                b_flip = b ^ (1 << i)
                b_flip ^= (1 << j)
                y_idx.append(b_flip)
                y_val.append(J * amp_b)

        # done with x_local
        x.restoreArray(x_local)

        # add all contributions to y
        y.setValues(y_idx, y_val, addv=PETSc.InsertMode.ADD_VALUES)

        # assemble the Vec
        y.assemblyBegin()
        y.assemblyEnd()


def build_hamiltonian_dodecahedron(comm, N, J, h):
    """
    Return a PETSc shell matrix H_shell that represents:

        H =  J * sum_{(i,j) in dodecahedral_bonds()} σ_i^x σ_j^x
           - h * sum_i σ_i^z

    in the computational Z basis (bitstrings of length N).

    This H_shell is what we'll feed to the Chebyshev solver.
    """
    ctx = DodecahedronHamiltonianShellCtx(N, J, h, comm)
    dim = ctx.dim

    H_shell = PETSc.Mat().create(comm=comm)
    H_shell.setType(PETSc.Mat.Type.SHELL)
    H_shell.setSizes([[dim, dim], [dim, dim]])
    H_shell.setUp()
    H_shell.setPythonContext(ctx)
    H_shell.setOption(PETSc.Mat.Option.SYMMETRIC, True)

    return H_shell


# -----------------------------------------------------------------------------
# 4. Chebyshev-filtered interior eigensolver
#
#    Basic outline:
#      - Affine-scale H to Ĥ with spectrum in [-1,1]
#      - Build a polynomial filter peaked near mu
#      - Iterate filter on a block subspace
#      - Rayleigh-Ritz on that subspace
# -----------------------------------------------------------------------------

def chebyshev_filter_apply(H_shifted, X, mu_hat, alpha, poly_deg, comm):
    """
    Apply a Gaussian-like Chebyshev polynomial filter centered at mu_hat
    to each column in the block X.

    Steps:
    - Approximate f(θ) = exp(-α (θ-μ̂)^2) on θ∈[-1,1] with Chebyshev series.
    - Evaluate via Clenshaw recurrence.
    """

    # ---- 1. Estimate Chebyshev coefficients c[m] up to poly_deg
    Msample = 200
    thetas = np.cos(np.pi * (np.arange(Msample) + 0.5) / Msample)  # in [-1,1]
    f_vals = np.exp(-alpha * (thetas - mu_hat)**2)
    tks = np.arccos(thetas)  # map back to [0,π]

    c = np.zeros(poly_deg + 1, dtype=np.float64)
    dt = np.pi / Msample
    for m in range(poly_deg + 1):
        cosmt = np.cos(m * tks)
        accum = np.sum(f_vals * cosmt)
        if m == 0:
            c[m] = accum * dt / np.pi
        else:
            c[m] = 2.0 * accum * dt / np.pi

    # ---- 2. Clenshaw recurrence (block version)
    def new_block_like(Xref):
        Ntot, nvec = Xref.getSize()
        Y = PETSc.Mat().createDense([Ntot, nvec], comm=comm)
        Y.setUp()
        Y.zeroEntries()
        return Y

    b_k  = new_block_like(X)
    b_k1 = new_block_like(X)
    Hbk  = new_block_like(X)

    for k in range(poly_deg, 0, -1):
        # Hbk = H_shifted * b_k
        H_shifted.matMult(b_k, Hbk)

        # b_new = 2 * Hbk - b_k1 + c[k]*X
        b_new = new_block_like(X)
        b_new.axpy(2.0, Hbk)
        b_new.axpy(-1.0, b_k1)
        if abs(c[k]) > 0:
            b_new.axpy(c[k], X)

        b_k1 = b_k
        b_k  = b_new

    result = new_block_like(X)
    H_shifted.matMult(b_k, result)
    result.axpy(-1.0, b_k1)
    result.axpy(0.5 * c[0], X)

    return result


def orthonormalize_block(X, comm):
    """
    Modified Gram-Schmidt on PETSc dense block X, in place.

    NOTE:
    - This implementation assumes single-rank (COMM_WORLD size == 1).
      In MPI >1, dot() must still work (it does global reductions),
      but extracting columns with getVecLeft() is fine.
    """
    Ntot, nvec = X.getSize()

    cols = []
    for j in range(nvec):
        v = X.getVecLeft(j)
        for q in cols:
            alpha = v.dot(q)
            v.axpy(-alpha, q)
        nrm = v.norm()
        if nrm > 1e-14:
            v.scale(1.0 / nrm)
        cols.append(v.copy())

    # write columns back
    for j, v in enumerate(cols):
        X.setVecLeft(j, v)

    X.assemblyBegin()
    X.assemblyEnd()
    return X


def rayleigh_ritz(H, X, comm):
    """
    Project H onto span(X) and solve the reduced eigenproblem.
    Returns Ritz values and Ritz vectors in the full space.
    """
    Ntot, k = X.getSize()
    tmp = H.createVecRight()
    H_sub = np.zeros((k, k), dtype=np.float64)

    for j in range(k):
        xj = X.getVecLeft(j)
        H.mult(xj, tmp)
        for i in range(k):
            xi = X.getVecLeft(i)
            H_sub[i, j] = xi.dot(tmp)

    w, Vsmall = np.linalg.eigh(H_sub)

    ritz_vecs_full = []
    for i in range(k):
        coeffs = Vsmall[:, i]
        v_full = H.createVecRight()
        v_full.set(0.0)
        for col in range(k):
            xc = X.getVecLeft(col)
            v_full.axpy(coeffs[col], xc)
        nrm = v_full.norm()
        if nrm > 1e-14:
            v_full.scale(1.0 / nrm)
        ritz_vecs_full.append(v_full)

    return w, ritz_vecs_full


def chebyshev_interior_solver(
    H_petsc,
    Emin=-62.0,
    Emax=62.0,
    mu=-18.0,
    nev=6,
    num_vecs=18,
    poly_deg=120,
    alpha=30.0,
    n_iter=10,
    seed=1234,
    comm=PETSc.COMM_WORLD,
):
    """
    Chebyshev-filtered subspace iteration to approximate interior eigenpairs.
    """

    rank = comm.getRank()

    # global dimension
    Ntot, _ = H_petsc.getSize()

    # Affine scaling H -> Ĥ whose spectrum is in [-1,1]
    a = 0.5 * (Emax - Emin)
    b = 0.5 * (Emax + Emin)

    class ShiftScaleCtx:
        def __init__(self, H, a, b):
            self.H = H
            self.a = a
            self.b = b
            self.tmp = H.createVecRight()

        def mult(self, A, x, y):
            # y = (H - b I) x / a
            self.H.mult(x, y)
            y.axpy(-self.b, x)
            y.scale(1.0 / self.a)

    shift_ctx = ShiftScaleCtx(H_petsc, a, b)

    H_shifted = PETSc.Mat().create(comm=comm)
    H_shifted.setType(PETSc.Mat.Type.SHELL)
    H_shifted.setSizes(H_petsc.getSizes())
    H_shifted.setUp()
    H_shifted.setPythonContext(shift_ctx)
    H_shifted.setOption(PETSc.Mat.Option.SYMMETRIC, True)

    mu_hat = (mu - b) / a

    # Initial random block X \in R^{Ntot x num_vecs}
    rng = np.random.default_rng(seed + rank)
    X = PETSc.Mat().createDense([Ntot, num_vecs], comm=comm)
    X.setUp()
    for j in range(num_vecs):
        v = H_petsc.createVecRight()
        arr = v.getArray()
        arr[:] = rng.standard_normal(size=arr.size)
        v.restoreArray(arr)
        X.setVecLeft(j, v)
    X.assemblyBegin()
    X.assemblyEnd()

    X = orthonormalize_block(X, comm)

    # Filter / orthonormalize loop
    for _ in range(n_iter):
        Y = chebyshev_filter_apply(H_shifted, X, mu_hat, alpha, poly_deg, comm)
        X = Y
        X = orthonormalize_block(X, comm)

    # Rayleigh-Ritz
    ritz_vals, ritz_vecs = rayleigh_ritz(H_petsc, X, comm)

    # Pick the closest nev to mu
    idx_sorted = np.argsort(np.abs(ritz_vals - mu))
    idx_keep = idx_sorted[:nev]
    eigvals_out = ritz_vals[idx_keep]
    eigvecs_out = [ritz_vecs[i] for i in idx_keep]

    # Sort picked ones by energy
    order = np.argsort(eigvals_out)
    eigvals_out = eigvals_out[order]
    eigvecs_out = [eigvecs_out[i] for i in order]

    return eigvals_out, eigvecs_out


# -----------------------------------------------------------------------------
# 5. main()
# -----------------------------------------------------------------------------

def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    # ----------------- PHYSICAL MODEL PARAMETERS (EDIT HERE) -----------------
    N_SPINS      = 20     # the dodecahedron graph has 20 sites
    COUPLING_J   = 1.0    # J in  J * Σ X_i X_j
    FIELD_h      = 1.0    # h in -h * Σ Z_i
    TARGET_MU    = -18.0  # target interior energy
    SPECTRUM_MIN = -62.0  # lower bound of total spectrum
    SPECTRUM_MAX =  62.0  # upper bound of total spectrum

    # --------------- CHEBYSHEV SOLVER TUNING (EDIT IF NEEDED) ----------------
    NEV        = 6        # how many eigenvalues to return near TARGET_MU
    NUM_VECS   = 18       # subspace/block size
    POLY_DEG   = 120      # Chebyshev polynomial degree
    ALPHA      = 30.0     # Gaussian sharpness in filter
    N_ITER     = 10       # filter / orthonormalize sweeps

    # ------------------------ BUILD H (PETSc SHELL) --------------------------
    # This H_shell represents exactly YOUR Hamiltonian:
    #   H =  J Σ X_i X_j  -  h Σ Z_i
    # on the 20-site dodecahedron.
    H_shell = build_hamiltonian_dodecahedron(
        comm,
        N=N_SPINS,
        J=COUPLING_J,
        h=FIELD_h,
    )

    # ----------------- RUN INTERIOR CHEBYSHEV SOLVER -------------------------
    eigvals, eigvecs = chebyshev_interior_solver(
        H_shell,
        Emin=SPECTRUM_MIN,
        Emax=SPECTRUM_MAX,
        mu=TARGET_MU,
        nev=NEV,
        num_vecs=NUM_VECS,
        poly_deg=POLY_DEG,
        alpha=ALPHA,
        n_iter=N_ITER,
        comm=comm,
    )

    # ----------------------- PRINT RESULTS (RANK 0) --------------------------
    if rank == 0:
        print(f"Target interior energy ~ {TARGET_MU}")
        print("Eigenvalues closest to target:")
        for i, val in enumerate(eigvals):
            print(f"{i:2d}: {val:.12f}")


if __name__ == "__main__":
    main()
