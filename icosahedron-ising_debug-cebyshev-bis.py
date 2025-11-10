"""
TFIM on the aicosahedron in PETSc/SLEPc form.

This file builds sparse PETSc matrices for:
    1. Full transverse-field Ising:
        H = J * sum_{<i,j>} σ^x_i σ^x_j  - h * sum_i σ^z_i
    2. Pure exchange (no field):
        H = J * sum_{<i,j>} σ^x_i σ^x_j
    3. Pure transverse field (no exchange):
        H = - h * sum_i σ^z_i

Geometry: 20-site dodecahedron.

All builders return PETSc.Mat in AIJ (CSR) format,
dimension 2^N × 2^N, suitable for SLEPc eigensolvers.

You can run a tiny sanity test at the bottom (guarded by __main__)
to check matrix size and a couple entries.
"""

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc


# ---------- Basic single-site Pauli matrices as numpy (2x2 dense) ----------

def pauli_x():
    """Pauli X matrix."""
    return np.array([[0.0, 1.0],
                     [1.0, 0.0]])

def pauli_z():
    """Pauli Z matrix."""
    return np.array([[1.0,  0.0],
                     [0.0, -1.0]])


# ---------- Lattice / molecular graph ----------

def icosahedral_bonds():
    """
    Connectivity (edges) of a 12-site icosahedral structure.

    Returns:
        list[tuple[int,int]]: each (i,j) is a bond between spins i and j.
    """
    bonds = [
        (0, 2), (0, 4), (0, 5), (0, 8), (0, 9),
        (1, 3), (1, 6), (1, 7), (1, 10), (1, 11),
        (2, 6), (2, 7), (2, 8), (2, 9), (3, 4),
        (3, 5), (3, 10), (3, 11), (4, 5), (4, 8),
        (4, 10), (5, 9), (5, 11), (6, 7), (6, 8),
        (6, 10), (7, 9), (7, 11), (8, 10), (9, 11)
    ]
    return bonds

# ---------- Local operator application helpers ----------

def add_sigma_z(mat, N, site, coeff):
    """
    Add coeff * σ^z(site) into PETSc matrix `mat`.

    σ^z on a single qubit is diagonal in the computational basis:
        |0> -> +1 |0>
        |1> -> -1 |1>

    Args:
        mat   (PETSc.Mat): matrix we're filling (2^N × 2^N).
        N     (int): number of spins.
        site  (int): which spin (0 ... N-1).
        coeff (float): prefactor.
    """
    dim = 1 << N  # 2**N
    for state in range(dim):
        bit = (state >> site) & 1
        val = +1.0 if bit == 0 else -1.0
        mat.setValue(state, state, coeff * val, addv=True)


def add_sigma_x_sigma_x(mat, N, i, j, coeff):
    """
    Add coeff * σ^x(i) σ^x(j) into PETSc matrix `mat`.

    σ^x flips a qubit.
    So σ^x_i σ^x_j acting on |state> gives |state with bits i and j flipped>.

    Off-diagonal in general.

    Args:
        mat   (PETSc.Mat): matrix we're filling (2^N × 2^N).
        N     (int): number of spins.
        i, j  (int): spin indices (0 ... N-1).
        coeff (float): prefactor J.
    """
    dim = 1 << N
    for state in range(dim):
        flipped_i  = state ^ (1 << i)
        flipped_ij = flipped_i ^ (1 << j)
        # <state| σ^x_i σ^x_j |flipped_ij> = 1
        mat.setValue(state, flipped_ij, coeff, addv=True)


# ---------- PETSc Hamiltonian builders ----------

def build_matrix_tfim_icosahedral(N, J, h):
    """
    Full transverse-field Ising on the icosahedron:
        H = J * Σ_{(i,j) in bonds} σ^x_i σ^x_j  - h * Σ_i σ^z_i

    Args:
        N (int): number of spins (should be 12 here).
        J (float): interaction strength.
        h (float): transverse field strength.

    Returns:
        PETSc.Mat: sparse Hamiltonian (AIJ) of dimension 2^N × 2^N
    """
    if N != 12:
        raise ValueError("Icosahedral molecule expects N = 12 sites.")

    bonds = icosahedral_bonds()
    dim = 1 << N  # 2**N

    H = PETSc.Mat().create()
    H.setType(PETSc.Mat.Type.AIJ)    # CSR sparse
    H.setSizes([dim, dim])

    # Rough nnz guess per row: each row connects to ~ (#bonds + N) states.
    # Not exact but helps PETSc avoid too many reallocs.
    H.setPreallocationNNZ(len(bonds) + N)

    H.setUp()

    # Add J * σ^x_i σ^x_j for all bonds
    for (i, j) in bonds:
        add_sigma_x_sigma_x(H, N, i, j, J)

    # Add -h * σ^z_i for all sites
    for site in range(N):
        add_sigma_z(H, N, site, -h)

    H.assemblyBegin()
    H.assemblyEnd()
    return H


def build_matrix_ising_dodecahedral(N, J):
    """
    Pure exchange Ising (no transverse field):
        H = J * Σ_{(i,j) in bonds} σ^x_i σ^x_j
    """
    if N != 20:
        raise ValueError("Dodecahedral molecule expects N = 20 sites.")

    bonds = dodecahedral_bonds()
    dim = 1 << N

    H = PETSc.Mat().create()
    H.setType(PETSc.Mat.Type.AIJ)
    H.setSizes([dim, dim])

    # each row couples to ~#bonds off-diagonally
    H.setPreallocationNNZ(len(bonds))

    H.setUp()

    for (i, j) in bonds:
        add_sigma_x_sigma_x(H, N, i, j, J)

    H.assemblyBegin()
    H.assemblyEnd()
    return H


def build_matrix_field_dodecahedral(N, h):
    """
    Pure transverse field (diagonal in Z basis):
        H = - h * Σ_i σ^z_i
    """
    if N != 20:
        raise ValueError("Dodecahedral molecule expects N = 20 sites.")

    dim = 1 << N

    H = PETSc.Mat().create()
    H.setType(PETSc.Mat.Type.AIJ)
    H.setSizes([dim, dim])

    # purely diagonal: 1 nnz per row is enough
    H.setPreallocationNNZ(1)

    H.setUp()

    for site in range(N):
        add_sigma_z(H, N, site, -h)

    H.assemblyBegin()
    H.assemblyEnd()
    return H

def estimate_extreme_eigenvalues(H, iters=50, rng=None):
    """
    Crude power iteration to estimate largest and smallest eigenvalues
    of symmetric real H.

    Returns:
        Emin_est, Emax_est
    """
    if rng is None:
        rng = np.random.default_rng()

    # largest (most positive) eigenvalue
    v = H.createVecRight()
    v_arr = rng.standard_normal(v.getLocalSize())
    v.setArray(v_arr)
    v.normalize()

    Hv = v.duplicate()
    for _ in range(iters):
        H.mult(v, Hv)   # Hv = H v
        Hv_norm = Hv.norm()
        Hv.scale(1.0 / Hv_norm)
        v, Hv = Hv, v  # swap refs (cheap)

    # Rayleigh quotient ~ max eigenvalue
    tmp = v.duplicate()
    H.mult(v, tmp)
    num = v.dot(tmp)
    den = v.dot(v)
    Emax_est = num / den
    tmp.destroy()

    # smallest eigenvalue: run power iteration on -H
    v = H.createVecRight()
    v_arr = rng.standard_normal(v.getLocalSize())
    v.setArray(v_arr)
    v.normalize()

    Hv = v.duplicate()
    for _ in range(iters):
        H.mult(v, Hv)     # Hv = H v
        Hv.scale(-1.0)    # Hv = -H v
        Hv_norm = Hv.norm()
        Hv.scale(1.0 / Hv_norm)
        v, Hv = Hv, v

    tmp = v.duplicate()
    H.mult(v, tmp)
    num = v.dot(tmp)
    den = v.dot(v)
    Emin_est = num / den
    tmp.destroy()

    v.destroy()
    Hv.destroy()

    return Emin_est, Emax_est

def slepc_extreme(H, which="max", tol=1e-7, maxit=500):
    """
    Use SLEPc to compute one extreme eigenvalue of Hermitian H.
    which: "max" -> largest real, "min" -> smallest real
    """
    eps = SLEPc.EPS().create()
    eps.setOperators(H)
    eps.setProblemType(SLEPc.EPS.ProblemType.HEP)  # Hermitian eigenproblem
    if which == "max":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    elif which == "min":
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    else:
        raise ValueError("which must be 'max' or 'min'")
    eps.setDimensions(1, PETSc.DECIDE)   # 1 eigenpair
    eps.setTolerances(tol, maxit)
    eps.solve()

    nconv = eps.getConverged()
    if nconv < 1:
        raise RuntimeError("SLEPc did not converge an eigenpair for bounds")

    vr = H.createVecRight()
    vi = H.createVecRight()
    eig = eps.getEigenpair(0, vr, vi)
    vr.destroy(); vi.destroy()
    return eig


def get_bounds_with_slepc(H, tol=1e-7, maxit=500):
    Emin = slepc_extreme(H, "min", tol=tol, maxit=maxit)
    Emin = Emin.real
    Emax = slepc_extreme(H, "max", tol=tol, maxit=maxit)
    Emax = Emax.real
    if Emin > Emax:
        Emin, Emax = Emax, Emin
    return Emin, Emax

# ------ Chebyshev filter / polynomial approximation ----------

def apply_Htilde(H, v_in, v_out, c, d):
    """
    Compute v_out = ((H - c I)/d) * v_in
      - H: PETSc.Mat (Hermitian)
      - v_in, v_out: PETSc.Vec (same layout)
      - c, d: floats from affine scaling with spectrum bounds

    Steps:
      tmp = H * v_in
      tmp -= c * v_in
      v_out = (1/d) * tmp
    """
    # tmp = H * v_in
    tmp = v_in.duplicate()
    H.mult(v_in, tmp)

    # tmp = tmp - c * v_in
    tmp.axpy(-c, v_in)     # tmp ← tmp + (-c) * v_in

    # v_out = (1/d) * tmp
    v_out.zeroEntries()
    v_out.axpy(1.0/d, tmp)

    tmp.destroy()


def jackson_weights(m):
    """
    Jackson damping weights to reduce Gibbs oscillations
    for Chebyshev series (order m).
    Returns g_k for k=0..m.
    """
    k = np.arange(m+1, dtype=float)
    M = float(m+1)
    g = ((M - k) * np.cos(np.pi * k / M) + np.sin(np.pi * k / M) / np.tan(np.pi / M)) / M
    return g

def chebyshev_filter(H, Emin, Emax, target_E0, m, pad=0.05, use_jackson=True, rng=None):
    """
    Apply a Chebyshev cosine kernel centered at target_E0.

    - Safely rescales with padded interval:
        [Emin', Emax'] with Emin' = Emin - pad*width, Emax' = Emax + pad*width
      so that the true spectrum lies strictly in (-1,1).

    - Alpha_k = cos(k * arccos(x0)) with x0 = (E0 - c)/d.
    - Optional Jackson damping.

    Returns:
        filt (normalized PETSc.Vec), approx_E (Rayleigh quotient)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pad bounds to be safe
    width = Emax - Emin
    Emin_p = Emin - pad * width
    Emax_p = Emax + pad * width

    c = 0.5 * (Emax_p + Emin_p)
    d = 0.5 * (Emax_p - Emin_p)

    x0 = (target_E0 - c) / d
    # Clamp x0 into [-0.999999, 0.999999] to avoid acos domain issues
    x0 = float(np.clip(x0, -0.999999, 0.999999))
    theta0 = np.arccos(x0)

    alpha = np.cos(np.arange(m+1) * theta0)
    if use_jackson:
        g = jackson_weights(m)
        alpha = alpha * g

    # random start
    v0 = H.createVecRight()
    v0.setRandom()  # PETSc random
    v0.normalize()

    # t0, t1
    t0 = v0.copy()
    t1 = v0.duplicate()
    apply_Htilde(H, v0, t1, c, d)  # (H-cI)/d

    filt = v0.duplicate()
    filt.zeroEntries()
    filt.axpy(alpha[0], t0)
    filt.axpy(alpha[1], t1)

    tkm1 = t0
    tk   = t1

    for k in range(2, m+1):
        tkp1 = v0.duplicate()
        apply_Htilde(H, tk, tkp1, c, d)   # tkp1 = H~ tk
        tkp1.scale(2.0)
        tkp1.axpy(-1.0, tkm1)             # tkp1 = 2 H~ tk - tkm1
        filt.axpy(alpha[k], tkp1)

        tkm1.destroy()
        tkm1 = tk
        tk = tkp1

    # normalize and Rayleigh quotient
    filt.normalize()
    Hv = filt.duplicate()
    H.mult(filt, Hv)
    approx_E = filt.dot(Hv) / filt.dot(filt)

    # cleanup
    Hv.destroy(); tk.destroy(); tkm1.destroy(); t1.destroy(); t0.destroy(); v0.destroy()
    return filt, approx_E

def chebyshev_apply_to_vec(H, v_in, Emin, Emax, target_E0, m, pad=0.05,
                           use_jackson=True):
    width = Emax - Emin
    Emin_p = Emin - pad*width
    Emax_p = Emax + pad*width

    c = 0.5 * (Emax_p + Emin_p)
    d = 0.5 * (Emax_p - Emin_p)

    x0 = (target_E0 - c) / d
    x0 = float(np.clip(x0, -0.999999, 0.999999))
    theta0 = np.arccos(x0)

    alpha = np.cos(np.arange(m+1) * theta0)
    if use_jackson:
        g = jackson_weights(m)
        alpha *= g

    # t0, t1
    t0 = v_in.copy()
    t0.normalize()
    t1 = v_in.duplicate()
    apply_Htilde(H, t0, t1, c, d)  # (H - cI)/d

    filt = v_in.duplicate()
    filt.zeroEntries()
    filt.axpy(alpha[0], t0)
    filt.axpy(alpha[1], t1)

    tkm1 = t0
    tk   = t1

    for k in range(2, m+1):
        tkp1 = v_in.duplicate()
        apply_Htilde(H, tk, tkp1, c, d)
        tkp1.scale(2.0)
        tkp1.axpy(-1.0, tkm1)
        filt.axpy(alpha[k], tkp1)

        tkm1.destroy()
        tkm1 = tk
        tk = tkp1

    # cleanup
    tk.destroy(); tkm1.destroy()
    return filt

def orthonormalize_block(V, tol=0.0):
    Q = []
    for v in V:
        w = v.copy()
        for q in Q:
            coeff = q.dot(w)
            w.axpy(-coeff, q)
        nrm = w.norm()
        if nrm > tol:
            w.scale(1.0 / nrm)
            Q.append(w)
    return Q

def chebyshev_block(H, Emin, Emax, target_E0, m,
                    block_size=8, pad=0.05, use_jackson=False, rng=None):
    """
    Block Chebyshev filter around target_E0.

    Returns:
        theta: 1D numpy array of Ritz values (length p)
        ritz_vecs: list of PETSc.Vec (length p)
    """

    if rng is None:
        import numpy.random as npr
        rng = npr.default_rng()

    # 1. random initial block
    V = []
    for _ in range(block_size):
        v = H.createVecRight()
        v.setRandom()
        v.normalize()
        V.append(v)

    # 2. apply Chebyshev filter to each vector
    W = []
    for v in V:
        w = chebyshev_apply_to_vec(H, v, Emin, Emax, target_E0, m,
                                   pad=pad, use_jackson=use_jackson)
        W.append(w)
    print("len(W) before ortho:", len(W))

    # 3. orthonormalize -> Q
    Q = orthonormalize_block(W)
    p = len(Q)
    print("len(Q) after ortho:", p)
    if p == 0:
        raise RuntimeError("Block orthonormalization produced no vectors (all near-zero).")

    # 4. Build small matrix T_ij = <q_i, H q_j>
    T = np.zeros((p, p), dtype=np.float64)
    tmp = H.createVecRight()

    # sanity check: sizes
    N = H.getSize()[0]
    for q in Q:
        assert q.getSize() == N, "Vec size mismatch with H"

    for j in range(p):
        H.mult(Q[j], tmp)        # tmp = H q_j
        for i in range(p):
            T[i, j] = Q[i].dot(tmp)

    # 5. diagonalize T
    theta, Y = np.linalg.eigh(T)

    # 6. form Ritz vectors u_i = sum_j Q_j * Y[j,i]
    ritz_vecs = []
    for i in range(p):
        u = H.createVecRight()
        u.set(0.0)
        for j in range(p):
            if abs(Y[j, i]) != 0.0:
                u.axpy(Y[j, i], Q[j])
        # (optional) normalize to be safe
        nrm = u.norm()
        if nrm > 0:
            u.scale(1.0 / nrm)
        ritz_vecs.append(u)

    return theta, ritz_vecs


if __name__ == "__main__":
    
    ########################################################################
    # Full 12-site icosahedron build
    ########################################################################
    N_full = 12
    J_full = 1.0
    h_full = 3.0

    Emin_full = -37.9456425
    Emax_full =  41.28675302

    target_E0_full = -6.0

    print("\n=== Full icosahedron (N=12) ===")
    dim_full = 1 << N_full
    print("Hilbert space dimension 2^N =", dim_full)  # 4096

    H_full = build_matrix_tfim_icosahedral(N_full, J_full, h_full)
    print("H_full size:", H_full.getSize())
    print("H_full nnz: ", H_full.getInfo()['nz_used'])

    #check with slepc extreme
    Emin_fullb, Emax_fullb = get_bounds_with_slepc(H_full, tol=1e-7, maxit=500)
    print(f"SLEPc bounds: Emin = {Emin_fullb}, Emax = {Emax_fullb}")

    # Chebyshev filter of degree m around -6.0
    # m=80/100 is typical for fairly sharp focus; tweak as needed.
    m_poly = 3000

    filt_vec_full, approx_E_full = chebyshev_filter(
        H_full,
        Emin_full,
        Emax_full,
        target_E0_full,
        m=m_poly,
        pad=0.0,
        use_jackson=True
    )

    print(f"Filtered Rayleigh quotient near {target_E0_full}: {approx_E_full}")

    # Cleanup
    #filt_vec_full.destroy()
    #H_full.destroy()

    theta, ritz_vecs = chebyshev_block(H_full, Emin_full, Emax_full, target_E0=target_E0_full,
                                   m=3000, block_size=10, pad=0.01, use_jackson=True)

    print("Ritz values from block Chebyshev:")
    print(len(theta))
    print(theta)

    # pick the Ritz values closest to -6
    idx = np.argsort(np.abs(theta + 6.0))
    eigs_around_minus6 = theta[idx[:5]]
    vecs_around_minus6 = [ritz_vecs[i] for i in idx[:5]]
    print("Ritz values near -6.0:", eigs_around_minus6)