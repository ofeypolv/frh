#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#from numpy.linalg import norm
import os
from scipy.linalg import eigh, qr, null_space, norm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.sparse import eye, kron, identity, csr_matrix, csc_matrix, lil_matrix, dok_matrix, issparse, coo_matrix
from scipy.sparse.linalg import eigsh, eigs, lobpcg, LinearOperator, ArpackNoConvergence
from scipy.optimize import curve_fit
from qutip import Qobj, ptrace, entropy_vn, qeye, tensor
from tqdm import tqdm
from itertools import product
from functools import reduce
import torch
import torch.optim as optim
from torch.autograd import Variable
import sympy as sp
from collections import Counter
from IPython.display import display, HTML


# In[ ]:


def pauli_x():
    """Pauli X matrix."""
    return np.array([[0, 1], [1, 0]])

def pauli_z():
    """Pauli Z matrix."""
    return np.array([[1, 0], [0, -1]])

def dodecahedral_bonds(): #20 vertices
    """
    Defines the connectivity of a true 20-vertex dodecahedral molecular structure.

    Returns:
        list of tuples: Each tuple (i, j) represents a bond between spin i and spin j.
    """
    bonds = [(0, 13), (0, 14), (0, 15), (1, 4), (1, 5), (1, 12),
    (2, 6), (2, 13), (2, 18), (3, 7), (3, 14), (3, 19),
    (4, 10), (4, 18), (5, 11), (5, 19), (6, 10), (6, 15),
    (7, 11), (7, 15), (8, 9), (8, 13), (8, 16), (9, 14), (9, 17),
    (10, 11), (12, 16), (12, 17), (16, 18), (17, 19)]

    return bonds


def transverse_field_ising_dodecahedral(N, J, h):
    """
    Constructs the Hamiltonian for the transverse field Ising model on an dodecahedral molecular structure.

    Parameters:
        N (int): Number of spins (should match the dodecahedral molecule, typically N=20).
        J (float): Interaction strength.
        h (float): Transverse field strength.

    Returns:
        H (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse format.
    """
    if N != 20:
        raise ValueError("Dodecahedral molecules typically have N = 20 sites.")

    # Sparse identity matrix
    I = identity(2, format="csr")

    # Pauli matrices as sparse matrices
    X = csr_matrix(pauli_x())
    Z = csr_matrix(pauli_z())

    # Initialize the Hamiltonian
    H = csr_matrix((2**N, 2**N), dtype=np.float64)

    # Get dodecahedral bonds
    bonds = dodecahedral_bonds()

    # Interaction term: J * sigma_i^x * sigma_j^x for dodecahedral connectivity
    for i, j in bonds:
        term = 1
        for k in range(N):
            if k == i or k == j:
                term = kron(term, X, format="csr")
            else:
                term = kron(term, I, format="csr")
        H += J * term

    # Transverse field term: -h * sigma_i^z
    for i in range(N):
        term = 1
        for j in range(N):
            if j == i:
                term = kron(term, Z, format="csr")
            else:
                term = kron(term, I, format="csr")
        H += -h * term

    return H

def ising_dodecahedron(N, J):
    """
    Constructs the Hamiltonian for the transverse field Ising model on a dodecahedral molecular structure without transverse field.

    Parameters:
        N (int): Number of spins (should match the dodecahedral molecule, typically N=20).
        J (float): Interaction strength.
        """
    if N != 20:
        raise ValueError("Dodecahedral molecules typically have N = 20 sites.")
    # Sparse identity matrix
    I = identity(2, format="csr")

    # Pauli matrices as sparse matrices
    X = csr_matrix(pauli_x())

    # Initialize the Hamiltonian
    H = csr_matrix((2**N, 2**N), dtype=np.float64)

    # Get dodecahedral bonds
    bonds = dodecahedral_bonds()

    # Interaction term: J * sigma_i^x * sigma_j^x for dodecahedral connectivity
    for i, j in bonds:
        term = 1
        for k in range(N):
            if k == i or k == j:
                term = kron(term, X, format="csr")
            else:
                term = kron(term, I, format="csr")
        H += J * term

    return H

def transverse_field_dodecahedral(N, h):
    """
    Constructs the Hamiltonian for the transverse field Ising model on a dodecahedral molecular structure.

    Parameters:
        N (int): Number of spins (should match the dodecahedral molecule, typically N=20).
        J (float): Interaction strength.
        h (float): Transverse field strength.

    Returns:
        H (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse format.
    """
    if N != 20:
        raise ValueError("Dodecahedral molecules typically have N = 20 sites.")

    # Sparse identity matrix
    I = identity(2, format="csr")

    # Pauli matrices as sparse matrices
    Z = csr_matrix(pauli_z())

    # Initialize the Hamiltonian
    H = csr_matrix((2**N, 2**N), dtype=np.float64)

    # Get dodecahedral bonds
    bonds = dodecahedral_bonds()

    # Transverse field term: -h * sigma_i^z
    for i in range(N):
        term = 1
        for j in range(N):
            if j == i:
                term = kron(term, Z, format="csr")
            else:
                term = kron(term, I, format="csr")
        H += -h * term

    return H

#######################################################################################################################

'''
def partial_trace_qubit(rho, keep, dims):
    """Compute the partial trace of a density matrix of qubits."""
    keep_dims = np.prod([dims[i] for i in keep])
    trace_dims = np.prod([dims[i] for i in range(len(dims)) if i not in keep])
    rho = rho.reshape([keep_dims, trace_dims, keep_dims, trace_dims])
    return np.trace(rho, axis1=1, axis2=3).reshape([keep_dims, keep_dims])

def partial_trace_qubit_torch(rho, keep, dims):
    """Compute the partial trace of a density matrix of qubits using PyTorch."""
    keep_dims = torch.prod(torch.tensor([dims[i] for i in keep]))
    trace_dims = torch.prod(torch.tensor([dims[i] for i in range(len(dims)) if i not in keep]))
    rho = rho.view(keep_dims, trace_dims, keep_dims, trace_dims)
    # Compute the partial trace
    traced_rho = torch.zeros((keep_dims, keep_dims), dtype=rho.dtype)
    for i in range(trace_dims):
        traced_rho += rho[:, i, :, i]
    #return traced_rho.view(keep_dims, keep_dims)
    return traced_rho'''

def isket_numpy(arr):
    """
    Check if a NumPy array is a ket (column vector).

    Parameters:
    - arr: np.ndarray, the array to check.

    Returns:
    - bool, True if the array is a ket, False otherwise.
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    shape = arr.shape

    if len(shape) == 2 and shape[1] == 1:
        return True
    else:
        return False

def ptrace_numpy(Q, sel, dims): # numpy function adapted from ptrace of qutip
    """
    Compute the partial trace of a density matrix of qubits using NumPy.

    Parameters:
    - Q: numpy object, the quantum object (density matrix or state vector).
    - sel: list of int, indices of the subsystems to keep.
    - dims: list of int, dimensions of the subsystems.

    Returns:
    - numpy object, the reduced density matrix after tracing out the specified subsystems.
    """
    # Get the dimensions of the subsystems
    rd = np.asarray(dims[0], dtype=np.int32).ravel()
    nd = len(rd)

    # Ensure sel is a sorted array of indices
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)
    sel = list(np.sort(sel))

    # Dimensions of the subsystems to keep
    dkeep = (rd[sel]).tolist()

    # Indices of the subsystems to trace out
    qtrace = list(set(np.arange(nd)) - set(sel))

    # Dimensions of the subsystems to trace out
    dtrace = (rd[qtrace]).tolist()

    # Reshape the density matrix or state vector
    rd = list(rd)
    if isket_numpy(Q):
        # Reshape and transpose for state vector
        vmat = (Q
                .reshape(rd)
                .transpose(sel + qtrace)
                .reshape([np.prod(dkeep), np.prod(dtrace)]))
        # Compute the reduced density matrix
        rhomat = vmat.dot(vmat.conj().T)
    else:
        # Reshape and transpose for density matrix
        rhomat = np.trace(Q
                          .reshape(rd + rd)
                          .transpose(qtrace + [nd + q for q in qtrace] +
                                     sel + [nd + q for q in sel])
                          .reshape([np.prod(dtrace),
                                    np.prod(dtrace),
                                    np.prod(dkeep),
                                    np.prod(dkeep)]))
    return rhomat


def ptrace_sparse(psi_sparse, keep, dims):
    """
    Compute the partial trace over arbitrary subsystems using sparse matrix operations.

    Args:
        psi_sparse (scipy.sparse matrix): Full density matrix of shape (D, D), where D = product(dims)
        keep (list of int): Subsystems to keep (indices, 0-indexed)
        dims (list of int): List of subsystem dimensions, e.g., [2]*n for n qubits

    Returns:
        scipy.sparse.csr_matrix: Reduced density matrix over kept subsystems
    """
    if not issparse(psi_sparse):
        raise ValueError("psi_sparse must be a scipy.sparse matrix")
    n = len(dims)
    D = np.prod(dims)
    if psi_sparse.shape != (D, D):
        raise ValueError("Density matrix shape does not match dims")
    trace = [i for i in range(n) if i not in keep]
    d_keep = np.prod([dims[i] for i in keep])
    # Prepare output
    data = []
    row_idx = []
    col_idx = []

    # Precompute bit masks
    def idx_to_bits(idx):
        return np.array(list(np.binary_repr(idx, width=n))).astype(int)


    psi_sparse = psi_sparse.tocoo()
    for i, j, val in zip(psi_sparse.row, psi_sparse.col, psi_sparse.data):
        bi = idx_to_bits(i)
        bj = idx_to_bits(j)


        # Only sum terms where traced-out subsystems agree
        if np.all(bi[trace] == bj[trace]):
            # Extract kept bits and convert to reduced indices
            #print('condition met for i, j:', i, j)
            i_red_bits = bi[keep]
            j_red_bits = bj[keep]
            i_red = int("".join(i_red_bits.astype(str)), 2)
            j_red = int("".join(j_red_bits.astype(str)), 2)


            data.append(val)
            row_idx.append(i_red)
            col_idx.append(j_red)

    return coo_matrix((data, (row_idx, col_idx)), shape=(d_keep, d_keep)).tocsr()


def isket_torch(arr):
    """
    Check if a PyTorch tensor is a ket (column vector).

    Parameters:
    - arr: torch.Tensor, the array to check.

    Returns:
    - bool, True if the array is a ket, False otherwise.
    """
    if not isinstance(arr, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    shape = arr.shape

    if len(shape) == 2 and shape[1] == 1:
        return True
    else:
        return False

def ptrace_torch(Q, sel, dims): # torch function adapted from ptrace of qutip
    """
    Compute the partial trace of a density matrix of qubits using PyTorch.

    Parameters:
    - Q: torch.Tensor, the quantum object (density matrix or state vector).
    - sel: list of int, indices of the subsystems to keep.
    - dims: list of int, dimensions of the subsystems.

    Returns:
    - torch.Tensor, the reduced density matrix after tracing out the specified subsystems.
    """
    # Get the dimensions of the subsystems
    rd = torch.tensor(dims[0], dtype=torch.int32).flatten()
    nd = len(rd)
    #print("rd", rd)
    #print("nd", nd)

    # Ensure sel is a sorted array of indices
    if isinstance(sel, int):
        sel = torch.tensor([sel])
    else:
        sel = torch.tensor(sel)
    sel = torch.sort(sel).values.tolist()

    # Dimensions of the subsystems to keep
    dkeep = rd[sel].tolist()

    # Indices of the subsystems to trace out
    qtrace = list(set(range(nd)) - set(sel))

    # Dimensions of the subsystems to trace out
    dtrace = rd[qtrace].tolist()

    # Reshape the density matrix or state vector
    rd = rd.tolist()
    if isket_torch(Q):
        # Reshape and transpose for state vector
        reshaped_Q = Q.reshape(rd)
        #print(reshaped_Q.shape)
        transposed_Q = reshaped_Q.permute(sel + qtrace)
        #print(transposed_Q.shape)
        vmat = transposed_Q.reshape([torch.prod(torch.tensor(dkeep)), torch.prod(torch.tensor(dtrace))])
        #print(vmat.shape)
        # Compute the reduced density matrix
        rhomat = vmat @ vmat.conj().T
        #print(rhomat.shape)
    else:
        # Reshape and transpose for density matrix
        reshaped_Q = Q.reshape(rd + rd)
        #print("reshaped_Q", reshaped_Q.shape)
        transposed_Q = reshaped_Q.permute(qtrace + [nd + q for q in qtrace] + sel + [nd + q for q in sel])
        #print("transposed_Q", transposed_Q.shape)
        reshaped_transposed_Q = transposed_Q.reshape([torch.prod(torch.tensor(dtrace)), torch.prod(torch.tensor(dtrace)), torch.prod(torch.tensor(dkeep)), torch.prod(torch.tensor(dkeep))])
        #print("reshaped_transposed_Q", reshaped_transposed_Q.shape)
        #rhomat = torch.trace(reshaped_transposed_Q)
        rhomat = torch.einsum('iikl->kl', reshaped_transposed_Q)
        # Trace out the first two dimensions
        #rhomat = torch.zeros((torch.prod(torch.tensor(dkeep)), torch.prod(torch.tensor(dkeep))), dtype=Q.dtype)
        #for i in range(reshaped_transposed_Q.shape[0]):
        #    for j in range(reshaped_transposed_Q.shape[1]):
        #        rhomat += reshaped_transposed_Q[i, j, :, :]
        #print("rhomat", rhomat.shape)
    return rhomat

def entanglement_entropy(psi, subsystem, total_size):

    '''Computes the bipartite entanglement entropy of a pure state.

    Parameters:
    psi : np.array
        The wavefunction (state vector) of the full system.
    subsystem_size : int
        The number of qubits in subsystem A.
    total_size : int
        The total number of qubits in the system.

    Returns:
    float
        The von Neumann entanglement entropy S_A.'''

    psi_matrix =  np.outer(psi, psi.conj())

    # Compute the reduced density matrix rho_A = Tr_B(|psi><psi|)
    rho_A = ptrace_numpy(psi_matrix, subsystem, [[2]*total_size, [2]*total_size])  # Partial trace over B

    # Compute eigenvalues of rho_A
    eigenvalues = np.linalg.eigvalsh(rho_A)

    # Filter out zero eigenvalues to avoid numerical issues in log calculation
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Compute von Neumann entropy S_A = -Tr(rho_A log rho_A)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy

def entanglement_entropy_torch(psi, subsystem, total_size):
    """
    Computes the bipartite entanglement entropy of a pure state using PyTorch.

    Parameters:
    - psi: torch.Tensor (complex), the wavefunction (state vector) of the full system.
    - subsystem_size: int, the number of qubits in subsystem A.
    - total_size: int, the total number of qubits in the system.

    Returns:
    - torch.Tensor (scalar), the von Neumann entanglement entropy S_A.
    """

    if not isinstance(psi, torch.Tensor):
        psi = torch.tensor(psi, dtype=torch.complex64)

    # Ensure psi is normalized
    psi = psi / torch.norm(psi)

    # Compute the density matrix |psi><psi|
    psi_matrix = torch.outer(psi, psi.conj())

    # Compute the reduced density matrix rho_A = Tr_B(|psi><psi|)
    rho_A = ptrace_torch(psi_matrix, subsystem, [[2] * total_size, [2] * total_size])  # Partial trace over B

    #rho_A = rho_A.to(dtype=torch.float64)

    # Compute eigenvalues of rho_A
    eigvals = torch.linalg.eigvalsh(rho_A)

    # Filter out zero eigenvalues to avoid numerical issues in log calculation
    eigvals = eigvals[eigvals > 0]

    # Compute von Neumann entropy S_A = -Tr(rho_A log rho_A)
    entropy = -torch.sum(eigvals * torch.log2(eigvals))

    return entropy

def entanglement_entropy_qutip(psi, subsystem, total_size):

    # Convert the wavefunction to a QuTiP Qobj
    density_matrix = np.outer(psi, psi.conj())
    density_matrix_qobj = Qobj(density_matrix, dims=[[2]*total_size, [2]*total_size])

    rho_A = ptrace(density_matrix_qobj, subsystem)
    # Compute the von Neumann entropy S_A
    entropy = entropy_vn(rho_A, base=2)

    return entropy

def entanglement_entropy_np_ptrace(rdm):
    # rdm already computed and converted to numpy
    # Compute eigenvalues of rho_A
    eigenvalues = np.linalg.eigvalsh(rdm)

    # Filter out zero eigenvalues to avoid numerical issues in log calculation
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Compute von Neumann entropy S_A = -Tr(rho_A log rho_A)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy

def entanglement_entropy_torch_ptrace(rdm):

    eigvals = torch.linalg.eigvalsh(rdm)
    eigvals = eigvals[eigvals > 0]
    entropy = -torch.sum(eigvals * torch.log2(eigvals))
    return entropy


def entanglement_entropy_qutip_torch(psi, N):
    """
    Compute the von Neumann entanglement entropy using qutip.

    Parameters:
    - psi: torch.Tensor (complex), state vector of a quantum system.
    - N: int, total number of qubits.

    Returns:
    - torch.Tensor (scalar), von Neumann entropy.
    """
    # Ensure psi is normalized
    psi = psi / torch.norm(psi)

    # Convert PyTorch tensor to NumPy for QuTiP
    psi_np = psi.detach().numpy()

    rho_np = np.outer(psi_np, psi_np.conj())
    rho_qobj = Qobj(rho_np, dims=[[2] * N, [2] * N])

    rho_A = ptrace(rho_qobj, list(range(N // 2)))

    # Compute von Neumann entropy
    entropy = entropy_vn(rho_A, base=2)  # Compute in log base 2

    # Convert back to PyTorch tensor to allow gradient flow
    return torch.tensor(entropy, dtype=torch.float32, requires_grad=True)

#######################################################################################################################

# Define the linear combination function - numpy
def linear_combination_np(coeffs, psis):
    # Ensure psis are numpy tensors
    psi_np = [np.array(psi) for psi in psis]
    # Compute the linear combination in PyTorch
    psi = sum(c * psi for c, psi in zip(coeffs, psis))

    return psi

# Define the linear combination function - torch
def linear_combination(coeffs, psis):
    # Ensure psis are PyTorch tensors
    psis_torch = [torch.tensor(psi, dtype=torch.complex64) if not isinstance(psi, torch.Tensor) else psi for psi in psis]

    # Compute the linear combination in PyTorch
    psi_torch = sum(c * psi for c, psi in zip(coeffs, psis_torch))

    return psi_torch

# Define the linear combination function - torch but after computing the ptrace of outer products of scars
def linear_combination_outer(coeffs, outs):
    # Ensure outs are PyTorch tensors
    outs_torch = [torch.tensor(out, dtype=torch.complex64) if not isinstance(out, torch.Tensor) else out for out in outs]
    torch_coeffs = torch.tensor(coeffs, dtype=torch.complex64)

    # Compute the PyTorch tensor of out_coeffs which is the product of all possible combinations of c_i^* times c_j
    out_coeffs = torch.zeros((len(torch_coeffs), len(torch_coeffs)), dtype=torch.complex64)
    for i in range(len(torch_coeffs)):
        for j in range(len(torch_coeffs)):
            out_coeffs[i, j] = torch.conj(torch_coeffs[i]) * torch_coeffs[j]

    # Compute the linear combination in PyTorch
    lin_torch = sum(out_coeffs[i, j] * outs_torch[i] for i in range(len(coeffs)) for j in range(len(coeffs)))

    return lin_torch

######################################################

# chebyshev

def jackson_weights(m):
    """
    Jackson damping coefficients for k = 0..m.
    (You can replace this with your own implementation if you already have one.)
    """
    k = np.arange(m+1, dtype=float)
    N = m + 1.0
    # Standard Jackson kernel for Chebyshev series
    # g_k = [(N - k + 1) * cos(pi*k/(N+1)) + sin(pi*k/(N+1)) / tan(pi/(N+1))] / (N+1)
    gk = ((N - k + 1) * np.cos(np.pi * k / (N + 1.0)) +
          np.sin(np.pi * k / (N + 1.0)) / np.tan(np.pi / (N + 1.0))) / (N + 1.0)
    return gk

def chebyshev_filter_numpy(H, Emin, Emax, target_E0, m,
                           pad=0.05, use_jackson=True, rng=None):
    """
    Chebyshev cosine kernel filter, pure NumPy/SciPy version.

    Parameters
    ----------
    H : (n, n) array_like or sparse_matrix
        Real symmetric / Hermitian matrix.
    Emin, Emax : float
        Estimated spectral bounds of H.
    target_E0 : float
        Target energy where we want to focus the filter.
    m : int
        Polynomial degree.
    pad : float, optional
        Padding fraction for bounds.
    use_jackson : bool, optional
        Apply Jackson damping.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    filt : ndarray, shape (n,)
        Normalized filtered vector.
    approx_E : float
        Rayleigh quotient <filt|H|filt>.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Padded bounds and rescaling parameters
    width  = Emax - Emin
    Emin_p = Emin - pad * width
    Emax_p = Emax + pad * width

    c = 0.5 * (Emax_p + Emin_p)
    d = 0.5 * (Emax_p - Emin_p)

    # 2) Rescaled target x0 and Chebyshev coefficients alpha_k
    x0 = (target_E0 - c) / d
    x0 = float(np.clip(x0, -0.999999, 0.999999))
    theta0 = np.arccos(x0)

    alpha = np.cos(np.arange(m+1) * theta0)
    if use_jackson:
        g = jackson_weights(m)
        alpha = alpha * g

    # Helper: matvec with Htilde = (H - c I)/d
    def Htilde_dot(v):
        Hv = H @ v   # works for dense or sparse
        return (Hv - c * v) / d

    # 3) Random start vector
    n = H.shape[0]
    v0 = rng.standard_normal(n)
    v0 /= norm(v0)

    # 4) Chebyshev recursion
    t0 = v0
    t1 = Htilde_dot(v0)

    filt = alpha[0] * t0 + alpha[1] * t1

    tkm1 = t0
    tk   = t1

    for k in range(2, m+1):
        tkp1 = 2.0 * Htilde_dot(tk) - tkm1
        filt = filt + alpha[k] * tkp1
        tkm1, tk = tk, tkp1

    # 5) Normalize and Rayleigh quotient
    filt_norm = norm(filt)
    if filt_norm == 0:
        raise RuntimeError("Filtered vector became zero; try different parameters.")
    filt /= filt_norm

    Hv = H @ filt
    approx_E = np.vdot(filt, Hv).real / np.vdot(filt, filt).real

    return filt, approx_E

def chebyshev_filter_v0_numpy(H, v0, Emin, Emax, target_E0, m,
                           pad=0.05, use_jackson=True, rng=None):
    """
    Chebyshev cosine kernel filter, pure NumPy/SciPy version.

    Parameters
    ----------
    H : (n, n) array_like or sparse_matrix
        Real symmetric / Hermitian matrix.
    v0 : (n,) array_like
        Initial vector to start the Chebyshev recursion.
    Emin, Emax : float
        Estimated spectral bounds of H.
    target_E0 : float
        Target energy where we want to focus the filter.
    m : int
        Polynomial degree.
    pad : float, optional
        Padding fraction for bounds.
    use_jackson : bool, optional
        Apply Jackson damping.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    filt : ndarray, shape (n,)
        Normalized filtered vector.
    approx_E : float
        Rayleigh quotient <filt|H|filt>.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) Padded bounds and rescaling parameters
    width  = Emax - Emin
    Emin_p = Emin - pad * width
    Emax_p = Emax + pad * width

    c = 0.5 * (Emax_p + Emin_p)
    d = 0.5 * (Emax_p - Emin_p)

    # 2) Rescaled target x0 and Chebyshev coefficients alpha_k
    x0 = (target_E0 - c) / d
    x0 = float(np.clip(x0, -0.999999, 0.999999))
    theta0 = np.arccos(x0)

    alpha = np.cos(np.arange(m+1) * theta0)
    if use_jackson:
        g = jackson_weights(m)
        alpha = alpha * g

    # Helper: matvec with Htilde = (H - c I)/d
    def Htilde_dot(v):
        Hv = H @ v   # works for dense or sparse
        return (Hv - c * v) / d

    # 3) Normalize random start vector if not already normalized
    v0 /= norm(v0)

    # 4) Chebyshev recursion
    t0 = v0
    t1 = Htilde_dot(v0)

    filt = alpha[0] * t0 + alpha[1] * t1

    tkm1 = t0
    tk   = t1

    for k in range(2, m+1):
        tkp1 = 2.0 * Htilde_dot(tk) - tkm1
        filt = filt + alpha[k] * tkp1
        tkm1, tk = tk, tkp1

    # 5) Normalize and Rayleigh quotient
    filt_norm = norm(filt)
    if filt_norm == 0:
        raise RuntimeError("Filtered vector became zero; try different parameters.")
    filt /= filt_norm

    Hv = H @ filt
    approx_E = np.vdot(filt, Hv).real / np.vdot(filt, filt).real

    return filt, approx_E

def chebyshev_filter_block_numpy(H, V0, Emin, Emax, target_E0, m,
                                 pad=0.05, use_jackson=True):
    """
    Block Chebyshev cosine kernel filter (pure NumPy/SciPy version).

    Parameters
    ----------
    H : (n, n) array_like or sparse matrix
        Real symmetric / Hermitian matrix.
    V0 : (n, p) array_like
        Initial block of p vectors (columns) to start the Chebyshev recursion.
        Columns should be linearly independent; they need not be orthonormal.
    Emin, Emax : float
        Estimated spectral bounds of H.
    target_E0 : float
        Target energy where we want to focus the filter.
    m : int
        Polynomial degree.
    pad : float, optional
        Padding fraction for bounds.
    use_jackson : bool, optional
        Apply Jackson damping to the Chebyshev coefficients.

    Returns
    -------
    Phi : ndarray, shape (n, p)
        Approximate eigenvectors (columns) near target_E0.
    evals : ndarray, shape (p,)
        Corresponding Ritz eigenvalues.
    """

    V0 = np.array(V0, dtype=np.complex128, copy=True)
    n, p = V0.shape

    # 1) Padded bounds and rescaling parameters
    width  = Emax - Emin
    Emin_p = Emin - pad * width
    Emax_p = Emax + pad * width

    c = 0.5 * (Emax_p + Emin_p)
    d = 0.5 * (Emax_p - Emin_p)

    # 2) Rescaled target x0 and Chebyshev coefficients alpha_k
    x0 = (target_E0 - c) / d
    x0 = float(np.clip(x0, -0.999999, 0.999999))
    theta0 = np.arccos(x0)

    alpha = np.cos(np.arange(m+1) * theta0)
    if use_jackson:
        g = jackson_weights(m)   # assumed defined elsewhere
        alpha = alpha * g

    # Helper: Htilde = (H - c I)/d acting on a block
    def Htilde_dot_block(V):
        HV = H @ V              # works for dense or sparse
        return (HV - c * V) / d

    # 3) Orthonormalize starting block: V0 -> Q0
    #    (this gives us an orthonormal basis of the initial subspace)
    Q0, _ = np.linalg.qr(V0)    # (n, p), orthonormal columns

    # 4) Block Chebyshev recursion
    T0 = Q0                     # (n, p)
    T1 = Htilde_dot_block(Q0)   # (n, p)

    filt = alpha[0] * T0 + alpha[1] * T1

    Tkm1 = T0
    Tk   = T1

    for k in range(2, m+1):
        Tkp1 = 2.0 * Htilde_dot_block(Tk) - Tkm1
        filt = filt + alpha[k] * Tkp1
        Tkm1, Tk = Tk, Tkp1

    # 5) Orthonormalize the filtered block
    Q, _ = np.linalg.qr(filt)   # (n, p), orthonormal columns spanning filtered subspace

    # 6) Rayleigh–Ritz in the filtered subspace
    # H_sub is the projected matrix H in basis Q
    H_sub = Q.conj().T @ (H @ Q)    # (p, p)
    evals, U = np.linalg.eigh(H_sub)

    # 7) Lift Ritz eigenvectors back to full space
    Phi = Q @ U    # (n, p)

    return Phi, evals

### symmetry sectors -- Ih later

def check_magnetization_sector(vec, N, tol=1e-6): ### total magnetization is not conserved --- it only applies to scars
    """Check average magnetization of a state vector."""
    D = 1 << N
    mag_avg = 0.0
    for b in range(D):
        mag_avg += magnetization(b, N) * np.abs(vec[b])**2
    return mag_avg

def check_parity_sector(vec, N, tol=1e-6):
    """Check if vector is in even/odd parity sector."""
    D = 1 << N
    weight_even = sum(np.abs(vec[b])**2 for b in range(D) if parity(b, N) == 1)
    weight_odd = sum(np.abs(vec[b])**2 for b in range(D) if parity(b, N) == -1)
    if weight_even > 1.0 - tol:
        return "even"
    elif weight_odd > 1.0 - tol:
        return "odd"
    else:
        return f"mixed (even={weight_even:.4f}, odd={weight_odd:.4f})"

def build_parity_operator(N):
    """
    Build the parity operator as a matrix.
    Parity operator P|b> = (-1)^(number of 1s) |b>

    Parameters:
    - N: int, number of qubits

    Returns:
    - P_op: sparse matrix, parity operator (diagonal)
    """
    D = 1 << N
    # build diagonal entries as a small-memory 1D array
    diag = np.fromiter((1 if bin(b).count('1') % 2 == 0 else -1 for b in range(D)),
                       dtype=np.int8, count=D)
    # construct sparse diagonal directly (COO -> CSR) without making a dense (D,D) array
    rows = np.arange(D, dtype=np.int64)
    P_op = csr_matrix((diag.astype(np.int8), (rows, rows)), shape=(D, D))
    return P_op

def commutator_norm(A, B):
    """
    Compute the Frobenius norm of the commutator [A, B] = AB - BA.
    For sparse matrices, use sparse operations.

    Parameters:
    - A, B: matrices (dense or sparse)

    Returns:
    - float, ||[A, B]||_F
    """
    comm = A @ B - B @ A

    if issparse(comm):
        # For sparse matrices, compute Frobenius norm
        return np.sqrt(comm.multiply(comm.conj()).sum())
    else:
        # For dense matrices
        return np.linalg.norm(comm, 'fro')

def magnetization(bitstring, N):
    # Suppose spin up = 1, spin down = 0
    # Or adjust convention as needed
    n_up = bitstring.bit_count()
    n_down = N - n_up
    return n_up - n_down  # proportional to total Sz

def parity(bitstring, N):
    """
    Compute parity of a bitstring.
    Returns +1 for even number of up spins, -1 for odd.
    """
    n_up = bitstring.bit_count()
    return 1 if (n_up % 2 == 0) else -1


# In[3]:


N = 20  # Number of spins
J = 1.0  # Interaction strength
h = 3.0  # Transverse field strength # this is the value in the paper. maybe try  other values too, including the critical value one (h=J=1)

# Assuming transverse_field_ising is defined and returns a sparse Hermitian matrix
H = transverse_field_ising_dodecahedral(N, J, h)
Hi = ising_dodecahedron(N, J)
Htf = transverse_field_dodecahedral(N, h)

print(f"Hamiltonian shape: {H.shape}")
print(f"Non-zero elements in H: {H.nnz}")


# In[4]:


# Build symmetry operators
P_op = build_parity_operator(N)

# compute Frobenius norm of H (works for sparse or dense)
if issparse(H):
    H_norm = np.sqrt((H.multiply(H.conj())).sum())
else:
    H_norm = np.linalg.norm(H, 'fro')
print("Checking if H commutes with symmetry operators:\n")

# Check [H, P] = 0 (parity symmetry)
comm_parity = commutator_norm(H, P_op)
print(f"Parity:")
print(f"  ||[H, P]||_F = {comm_parity:.6e}")
print(f"  ||H||_F = {H_norm:.6e}")
print(f"  Relative error: {comm_parity / H_norm:.6e}")
print(f"  Commutes: {'YES' if comm_parity / H_norm < 1e-10 else 'NO'}\n")

# Additional check: verify P^2 = Identity
P_squared = P_op @ P_op
Id_op = eye(1 << N, dtype=np.complex128, format='csr')   # <-- use eye(), don't shadow identity()

# FIX: subtract the matrix Id_op (not the identity function) and handle sparse/dense norm
diff = P_squared - Id_op
if issparse(diff):
    P_identity_error = np.sqrt((diff.multiply(diff.conj())).sum())
else:
    P_identity_error = np.linalg.norm(diff, 'fro')
# ...existing code...
print(f"Operator properties:")
print(f"  ||P^2 - I||_F = {P_identity_error:.6e}")
print(f"  P is involutory: {'YES' if P_identity_error < 1e-10 else 'NO'}")


# In[5]:


# ============================================================
# 0. Dodecahedron bonds (current labelling)
# ============================================================

bonds = [(0, 13), (0, 14), (0, 15), (1, 4), (1, 5), (1, 12),
    (2, 6), (2, 13), (2, 18), (3, 7), (3, 14), (3, 19),
    (4, 10), (4, 18), (5, 11), (5, 19), (6, 10), (6, 15),
    (7, 11), (7, 15), (8, 9), (8, 13), (8, 16), (9, 14), (9, 17),
    (10, 11), (12, 16), (12, 17), (16, 18), (17, 19)]

bonds_set = {tuple(sorted(e)) for e in bonds}
print("Number of bonds:", len(bonds_set))

# ============================================================
# 1. Load Ih permutations from file
# ============================================================

def load_ih_permutations(path="ih_dodeca_permutations_0based.txt"):
    """
    Each line in ih_dodeca_permutations_0based.txt represents a permutation on vertices 0..19.
    """
    perms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = line.strip('[]').split(',')
            perms.append(np.array([int(x) for x in nums], dtype=int))
    return perms

perms = load_ih_permutations("ih_dodeca_permutations_0based.txt")
N_sites = len(perms[0])
print(f"Loaded {len(perms)} I_h permutations on {N_sites} sites.")

# ============================================================
# 2. Graph automorphism checker (sanity)
# ============================================================

def is_automorphism(perm, bonds_set):
    """
    Check if 'perm' is a graph automorphism of the icosahedron
    defined by bonds_set, i.e. maps edges to edges.
    """
    for i, j in bonds_set:
        ii, jj = perm[i], perm[j]
        if tuple(sorted((ii, jj))) not in bonds_set:
            return False
    return True

num_auto = sum(is_automorphism(p, bonds_set) for p in perms)
print("Number of automorphisms wrt your bonds:", num_auto)


# In[6]:


# ============================================================
# 0. Dodecahedron bonds (current labelling)
# ============================================================

bonds = [(0, 13), (0, 14), (0, 15), (1, 4), (1, 5), (1, 12),
    (2, 6), (2, 13), (2, 18), (3, 7), (3, 14), (3, 19),
    (4, 10), (4, 18), (5, 11), (5, 19), (6, 10), (6, 15),
    (7, 11), (7, 15), (8, 9), (8, 13), (8, 16), (9, 14), (9, 17),
    (10, 11), (12, 16), (12, 17), (16, 18), (17, 19)]

bonds_set = {tuple(sorted(e)) for e in bonds}
print("Number of bonds:", len(bonds_set))

# ============================================================
# 1. Load Ih permutations from file
# ============================================================

def load_ih_permutations(path="ih_dodeca_permutations_0based.txt"):
    """
    Each line in ih_dodeca_permutations_0based.txt represents a permutation on vertices 0..19.
    """
    perms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = line.strip('[]').split(',')
            perms.append(np.array([int(x) for x in nums], dtype=int))
    return perms

perms = load_ih_permutations("ih_dodeca_permutations_0based.txt")
N_sites = len(perms[0])
print(f"Loaded {len(perms)} I_h permutations on {N_sites} sites.")

# ============================================================
# 2. Graph automorphism checker (sanity)
# ============================================================

def is_automorphism(perm, bonds_set):
    """
    Check if 'perm' is a graph automorphism of the dodecahedron
    defined by bonds_set, i.e. maps edges to edges.
    """
    for i, j in bonds_set:
        ii, jj = perm[i], perm[j]
        if tuple(sorted((ii, jj))) not in bonds_set:
            return False
    return True

num_auto = sum(is_automorphism(p, bonds_set) for p in perms)
print("Number of automorphisms wrt your bonds:", num_auto)

# ============================================================
# 3. Permutation utilities & conjugacy classes of I_h
# ============================================================

def perm_compose(p, q):
    """
    Composition p ∘ q acting on indices [0..N-1]:
    result r satisfies r[i] = p[q[i]].
    """
    return p[q]

def perm_inverse(p):
    """Inverse permutation p^{-1}."""
    inv = np.empty_like(p)
    inv[p] = np.arange(len(p))
    return inv

def perm_order(p, max_iter=300):
    """Order of permutation p: smallest k>0 with p^k = identity."""
    n = len(p)
    e = np.arange(n)
    x = p.copy()
    k = 1
    while not np.array_equal(x, e):
        x = perm_compose(x, p)
        k += 1
        if k > max_iter:
            raise RuntimeError("Permutation order too large?")
    return k

def compute_conjugacy_classes(perms):
    """
    Compute conjugacy classes of the group represented by 'perms'
    via g -> h g h^{-1}.
    """
    perms_tuples = [tuple(p.tolist()) for p in perms]
    perm_dict = {pt: p for pt, p in zip(perms_tuples, perms)}

    unseen = set(perms_tuples)
    classes = []

    while unseen:
        rep_t = unseen.pop()
        rep = perm_dict[rep_t]
        current = set()

        for h_t, h in perm_dict.items():
            h_inv = perm_inverse(h)
            conj = perm_compose(h, perm_compose(rep, h_inv))
            conj_t = tuple(conj.tolist())
            if conj_t in perm_dict:
                current.add(conj_t)

        for ct in current:
            unseen.discard(ct)

        class_perms = [perm_dict[ct] for ct in current]
        classes.append(class_perms)

    return classes

classes = compute_conjugacy_classes(perms)
print(f"Found {len(classes)} conjugacy classes in new labelling.")
for i, cls in enumerate(classes):
    size = len(cls)
    order = perm_order(cls[0])
    print(f"class {i}: size={size:2d}, order={order:2d}")

# ============================================================
# 4. Build Hilbert-space operator U_g for a permutation g
#    (big-endian convention, site 0 = most significant bit)
# ============================================================

def build_symmetry_operator(N_spins, perm):
    """
    U |s_0 ... s_{N-1}> = |s_{perm(0)} ... s_{perm(N-1)}|
    with site 0 as the left-most tensor factor (MSB).
    """
    D = 1 << N_spins
    rows = np.empty(D, dtype=np.int64)
    cols = np.arange(D, dtype=np.int64)

    for b in range(D):
        # decode: big-endian, site 0 = most significant bit
        bits = [(b >> (N_spins - 1 - i)) & 1 for i in range(N_spins)]

        # permute sites
        permuted_bits = [bits[perm[i]] for i in range(N_spins)]

        # re-encode in the same big-endian convention
        b_prime = 0
        for i in range(N_spins):
            b_prime |= permuted_bits[i] << (N_spins - 1 - i)

        rows[b] = b_prime

    data = np.ones(D, dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(D, D))

def check_Ug(N_spins, Ug):
    """
    Sanity checks: shape, permutation structure, and unitarity Ug† Ug = I.
    """
    dim = 1 << N_spins

    # 1. Size
    assert Ug.shape == (dim, dim), f"Ug has wrong shape: {Ug.shape}"

    # 2. Permutation structure: exactly one nonzero per row and column
    nnz_per_row = Ug.getnnz(axis=1)
    nnz_per_col = Ug.getnnz(axis=0)
    assert np.all(nnz_per_row == 1), "Some rows do not have exactly one '1'"
    assert np.all(nnz_per_col == 1), "Some columns do not have exactly one '1'"

    # 3. Unitarity: Ug† Ug = I
    Id = eye(dim, dtype=np.complex128, format='csr')
    diff = (Ug.conj().T @ Ug) - Id
    print("Number of nonzeros in Ug†Ug - I:", diff.nnz)
    if diff.nnz != 0:
        raise ValueError("Ug is not unitary: Ug†Ug - I has nonzero entries")

    print("All tests passed for this Ug.")

# ============================================================
# 5. Build all Ugs, check a few, and collect traces
# ============================================================

N_spins = N_sites  # 12 for icosahedron
dim = 1 << N_spins

Ugs = [build_symmetry_operator(N_spins, perm) for perm in perms]

traces = []
print("\nChecking a few U_g for unitarity and permutation structure...")
for i, Ug in enumerate(Ugs):
    if i < 3:
        check_Ug(N_spins, Ug)
    tr = Ug.diagonal().sum()
    traces.append(int(round(float(np.real(tr)))))

unique_traces = sorted(set(traces))
print(f"\nNumber of distinct trace values: {len(unique_traces)}")
print(f"Distinct values: {unique_traces}")

counts = Counter(traces)
print("Counts per trace:")
for val in unique_traces:
    print(f"  trace={val}: {counts[val]}")

# ============================================================
# 6. Build class operators C_k = sum_{g in class_k} U_g
# ============================================================

def build_class_operators(N_spins, classes):
    """
    Build class operators C_k = sum_{g in class_k} U_g.
    Returns a list of sparse CSR matrices.
    """
    class_ops = []
    for class_perms in classes:
        U_sum = None
        for perm in class_perms:
            U_g = build_symmetry_operator(N_spins, perm)
            if U_sum is None:
                U_sum = U_g.astype(np.complex128, copy=True).tocsr()
            else:
                U_sum = U_sum + U_g.tocsr()
        class_ops.append(U_sum.tocsr())
    return class_ops

# ============================================================
# 7. Optional: commutator norm checks with H, Hi, Htf
# ============================================================

def comm_norm(A, B):
    """Sparse Frobenius norm of commutator [A,B]."""
    C = A @ B - B @ A
    return np.sqrt((C.multiply(C.conj())).sum())

# ============================================================
# 8. Compute class data: size, order, χ_red for each conjugacy class
# ============================================================

def trace_from_perm(perm):
    Ug = build_symmetry_operator(N_spins, perm)
    return float(Ug.diagonal().sum())

class_data = []
for idx, cls in enumerate(classes):
    size = len(cls)
    order = perm_order(cls[0])
    chi = trace_from_perm(cls[0])
    class_data.append({
        "idx": idx,
        "size": size,
        "order": order,
        "chi": chi,
    })

print("\nRaw class summary (index, size, order, chi):")
for cd in class_data:
    print(cd)

def find_by(size=None, order=None):
    return [cd["idx"] for cd in class_data
            if (size is None or cd["size"] == size)
            and (order is None or cd["order"] == order)]

print("\nBy (size, order):")
sizes_orders = {}
for cd in class_data:
    key = (cd["size"], cd["order"])
    sizes_orders.setdefault(key, []).append(cd["idx"])
for k, v in sizes_orders.items():
    print(f"  {k}: indices {v}")

# ============================================================
# 9. Automatically build Ih standard order from class_data
#    Standard order: (E, 12C5, 12C5^2, 20C3, 15C2, i, 12S10, 12(S10)^3, 20S6, 15σ)
# ============================================================

idx_E_list   = find_by(size=1,  order=1)
idx_i_list   = find_by(size=1,  order=2)
idx_C5_list  = find_by(size=12, order=5)
idx_S10_list = find_by(size=12, order=10)
idx_C3_list  = find_by(size=20, order=3)
idx_S6_list  = find_by(size=20, order=6)
idx_15_list  = find_by(size=15, order=2)

assert len(idx_E_list)   == 1, "Expected 1 identity class"
assert len(idx_i_list)   == 1, "Expected 1 inversion class"
assert len(idx_C5_list)  == 2, "Expected two 5-fold rotation classes"
assert len(idx_S10_list) == 2, "Expected two S10 classes"
assert len(idx_C3_list)  == 1, "Expected one 20C3 class"
assert len(idx_S6_list)  == 1, "Expected one 20S6 class"
assert len(idx_15_list)  == 2, "Expected two 15-element classes"

idx_E  = idx_E_list[0]
idx_i  = idx_i_list[0]
idx_C3 = idx_C3_list[0]
idx_S6 = idx_S6_list[0]
idx_C5_a, idx_C5_b     = idx_C5_list
idx_S10_a, idx_S10_b   = idx_S10_list
idx_15_a, idx_15_b     = idx_15_list

print("\nAutomatic identification:")
print(" E:  ", idx_E)
print(" i:  ", idx_i)
print(" C5: ", idx_C5_list)
print(" S10:", idx_S10_list)
print(" C3: ", idx_C3)
print(" S6: ", idx_S6)
print(" 15-classes (C2, σ) candidates:", idx_15_list)

# ============================================================
# 10. Character table of Ih (in standard column order)
# ============================================================

w  = 2.0 * np.pi / 5.0
c2 = np.cos(2*w)   # cos(4π/5)
c1 = np.cos(1*w)   # cos(2π/5)

chi_irreps = np.array([
    # E,   12C5,     12C5^2,   20C3,  15C2,  i,    12S10,    12(S10)^3, 20S6,  15σ
    [1,    1,        1,        1,     1,     1,    1,        1,        1,     1],       # Ag
    [3,   -2*c2,    -2*c1,     0,    -1,     3,   -2*c1,    -2*c2,     0,    -1],       # T1g
    [3,   -2*c1,    -2*c2,     0,    -1,     3,   -2*c2,    -2*c1,     0,    -1],       # T2g
    [4,   -1,       -1,        1,     0,     4,   -1,       -1,        1,     0],       # Gg
    [5,    0,        0,       -1,     1,     5,    0,        0,       -1,     1],       # Hg
    [1,    1,        1,        1,     1,    -1,   -1,       -1,       -1,    -1],       # Au
    [3,   -2*c2,    -2*c1,     0,    -1,    -3,    2*c1,     2*c2,     0,     1],       # T1u
    [3,   -2*c1,    -2*c2,     0,    -1,    -3,    2*c2,     2*c1,     0,     1],       # T2u
    [4,   -1,       -1,        1,     0,    -4,    1,        1,       -1,     0],       # Gu
    [5,    0,        0,       -1,     1,    -5,    0,        0,        1,    -1],       # Hu
], dtype=float)

irrep_labels = ["Ag", "T1g", "T2g", "Gg", "Hg",
                "Au", "T1u", "T2u", "Gu", "Hu"]

G_order = 120

# ============================================================
# 11. Distinguish C2 vs σ by trace, then try all combinations
#     for C5/C5² and S10/(S10)³
# ============================================================

# Distinguish C2 vs σ by trace:
# - σ (reflection) has 8 cycles → trace = 2^12 = 4096
# - C2 (rotation) has 6 cycles → trace = 2^10 = 1024

chi_15_a = trace_from_perm(classes[idx_15_a][0])
chi_15_b = trace_from_perm(classes[idx_15_b][0])

print(f"\nDistinguishing C2 vs σ by trace:")
print(f"  Class {idx_15_a}: χ_red = {chi_15_a}")
print(f"  Class {idx_15_b}: χ_red = {chi_15_b}")

if abs(chi_15_a - 4096) < 1e-6:
    idx_sigma = idx_15_a
    idx_C2 = idx_15_b
    print(f"  → Class {idx_15_a} is σ (trace=4096, 12 cycles)")
    print(f"  → Class {idx_15_b} is C2 (trace=1024, 10 cycles)")
elif abs(chi_15_b - 4096) < 1e-6:
    idx_sigma = idx_15_b
    idx_C2 = idx_15_a
    print(f"  → Class {idx_15_b} is σ (trace=4096, 12 cycles)")
    print(f"  → Class {idx_15_a} is C2 (trace=1024, 10 cycles)")
else:
    raise ValueError("Cannot identify σ by trace=4096")

# C5/C5² and S10/(S10)³ have same trace, so check consistency
chi_C5_a = trace_from_perm(classes[idx_C5_a][0])
chi_C5_b = trace_from_perm(classes[idx_C5_b][0])
print(f"\n12C5 classes (same trace, check consistency):")
print(f"  Class {idx_C5_a}: χ_red = {chi_C5_a}")
print(f"  Class {idx_C5_b}: χ_red = {chi_C5_b}")

chi_S10_a = trace_from_perm(classes[idx_S10_a][0])
chi_S10_b = trace_from_perm(classes[idx_S10_b][0])
print(f"\n12S10 classes (same trace, check consistency):")
print(f"  Class {idx_S10_a}: χ_red = {chi_S10_a}")
print(f"  Class {idx_S10_b}: χ_red = {chi_S10_b}")

def compute_multiplicities(class_sizes, chi_red):
    """Compute irrep multiplicities using character orthogonality."""
    multiplicities = []
    for i in range(len(irrep_labels)):
        chi_Gamma = chi_irreps[i]
        n_Gamma = (class_sizes * chi_red * chi_Gamma).sum() / G_order
        multiplicities.append(n_Gamma)
    return np.array(multiplicities, dtype=float)

# Try all possible assignments for C5/C5² and S10/(S10)³
solutions = []

for (idx_C5_1, idx_C5_2) in [(idx_C5_a, idx_C5_b), (idx_C5_b, idx_C5_a)]:
    for (idx_S10_1, idx_S10_2) in [(idx_S10_a, idx_S10_b), (idx_S10_b, idx_S10_a)]:
        print(f"\n{'='*60}")
        print(f"Trying assignment:")
        print(f"  C2 -> class {idx_C2} (trace={chi_15_a if idx_C2==idx_15_a else chi_15_b})")
        print(f"  σ  -> class {idx_sigma} (trace={chi_15_a if idx_sigma==idx_15_a else chi_15_b})")
        print(f"  12C5 -> class {idx_C5_1}")
        print(f"  12C5² -> class {idx_C5_2}")
        print(f"  12S10 -> class {idx_S10_1}")
        print(f"  12(S10)³ -> class {idx_S10_2}")

        ordered_indices = [
            idx_E,
            idx_C5_1,
            idx_C5_2,
            idx_C3,
            idx_C2,
            idx_i,
            idx_S10_1,
            idx_S10_2,
            idx_S6,
            idx_sigma,
        ]

        class_sizes = np.zeros(10, dtype=int)
        chi_red = np.zeros(10, dtype=float)
        for k, class_idx in enumerate(ordered_indices):
            cls = classes[class_idx]
            class_sizes[k] = len(cls)
            rep_perm = cls[0]
            chi_red[k] = trace_from_perm(rep_perm)

        print("Class sizes:", class_sizes)
        print("chi_red:", chi_red)

        mult = compute_multiplicities(class_sizes, chi_red)
        mult_rounded = np.round(mult).astype(int)
        max_dev = np.max(np.abs(mult - mult_rounded))

        print("Raw multiplicities:", mult)
        print("Rounded multiplicities:", mult_rounded)
        print("Max deviation from integer:", max_dev)

        dims_irreps = np.array([1, 3, 3, 4, 5, 1, 3, 3, 4, 5], dtype=int)
        dim_check = (mult_rounded * dims_irreps).sum()
        print("∑ n_Γ d_Γ =", dim_check, "(expect 2^N =", 1 << N_spins, ")")

        ok = (max_dev < 1e-6) and (dim_check == (1 << N_spins)) and np.all(mult_rounded >= 0)
        if ok:
            print("=> This assignment is CONSISTENT ✓")
            solutions.append({
                "idx_C2": idx_C2,
                "idx_sigma": idx_sigma,
                "idx_C5_1": idx_C5_1,
                "idx_C5_2": idx_C5_2,
                "idx_S10_1": idx_S10_1,
                "idx_S10_2": idx_S10_2,
                "ordered_indices": ordered_indices,
                "class_sizes": class_sizes,
                "chi_red": chi_red,
                "multiplicities": mult_rounded,
            })
        else:
            print("=> This assignment is NOT consistent ✗")


# In[7]:


print(f"Found {len(solutions)} consistent assignment(s)")

if not solutions:
    raise RuntimeError("No consistent assignment found.")
else:
    sol = solutions[2]  # solutions[1] is also valid

    print("\n=== Final assignment (C2/σ by trace, C5/S10 by consistency) ===")
    print(f"C2  class index: {sol['idx_C2']}")
    print(f"sigma class index: {sol['idx_sigma']}")
    print(f"12C5 class index: {sol['idx_C5_1']}")
    print(f"12C5² class index: {sol['idx_C5_2']}")
    print(f"12S10 class index: {sol['idx_S10_1']}")
    print(f"12(S10)³ class index: {sol['idx_S10_2']}")
    print("Class sizes (Ih order):", sol["class_sizes"])
    print("chi_red (Ih order):", sol["chi_red"])
    print("\nIrrep multiplicities:")
    for label, n in zip(irrep_labels, sol["multiplicities"]):
        print(f"  {label}: n = {n}")

    dims_irreps = np.array([1, 3, 3, 4, 5, 1, 3, 3, 4, 5], dtype=int)
    print("\nCheck ∑ n_Γ d_Γ =",
          (sol["multiplicities"] * dims_irreps).sum(),
          "  (should be 2^N =", 1 << N_spins, ")")

# ============================================================
# 12. Character table in our ordering
# ============================================================

class_labels = [
    "E", "12C5", "12C5^2", "20C3", "15C2",
    "i", "12S10", "12(S10)^3", "20S6", "15σ",
]

print("\n=== Full Ih character table (χ_Γ(C)) in our ordering ===")
header = "irrep \\ class".ljust(10) + "  " + "  ".join(f"{c:>9}" for c in class_labels)
print(header)
print("-" * len(header))

for i, label in enumerate(irrep_labels):
    row = [label.ljust(10)]
    for j in range(10):
        row.append(f"{chi_irreps[i, j]:9.4f}")
    print("  ".join(row))


# In[8]:


# Reprint Ih conjugacy classes using the final assignment from `sol` (no raw indices)

if 'sol' not in globals():
    raise RuntimeError("Missing `sol` (run the previous assignment cell first).")
if 'classes' not in globals() or 'perm_order' not in globals() or 'trace_from_perm' not in globals():
    raise RuntimeError("Missing prerequisites (`classes`, `perm_order`, `trace_from_perm`).")

# Canonical Ih column order (must match how `sol['ordered_indices']` was built)
if 'class_labels_Ih' not in globals():
    class_labels_Ih = ["E","12C5","12C5^2","20C3","15C2","i","12S10","12(S10)^3","20S6","15σ"]

ordered_indices = sol["ordered_indices"]
print(ordered_indices)

canonical_classes = []
for lbl, raw_idx in zip(class_labels_Ih, ordered_indices):
    cls = classes[raw_idx]
    size = len(cls)
    order = perm_order(cls[0])
    chi = int(round(float(trace_from_perm(cls[0]))))
    canonical_classes.append({"label": lbl, "size": size, "order": order, "chi": chi})

print("Ih conjugacy classes (final assignment):")
for c in canonical_classes:
    # pretty power-of-two print if applicable
    exp = None
    if c["chi"] > 0:
        from math import log2
        e = log2(c["chi"])
        if abs(e - round(e)) < 1e-12:
            exp = int(round(e))
    if exp is not None:
        print(f"{c['label']:12s} size={c['size']:2d}  order={c['order']:2d}  chi={c['chi']} (2^{exp})")
    else:
        print(f"{c['label']:12s} size={c['size']:2d}  order={c['order']:2d}  chi={c['chi']}")


# In[9]:


# Sparse-only check: verify [H, U_g] = 0 for all Ugs
tol_rel = 1e-10  # relative tolerance
if not issparse(H):
    raise RuntimeError("H must be sparse for the sparse-only check")

# Frobenius norm of H (sparse)
H_norm = float(np.sqrt(np.real((H.multiply(H.conj())).sum())))
if H_norm == 0:
    H_norm = 1.0

bad = []
rel_values = []
for i, Ug in enumerate(Ugs):
    # sparse commutator
    C = H @ Ug - Ug @ H
    # Frobenius norm via sparse elementwise multiply and sum
    s = (C.multiply(C.conj())).sum()
    nrm = float(np.sqrt(np.real(s)))
    rel = nrm / H_norm
    rel_values.append((i, nrm, rel))
    if rel > tol_rel:
        bad.append((i, nrm, rel))

# summary
rel_values_sorted = sorted(rel_values, key=lambda x: x[2], reverse=True)
print(f"H Frobenius norm (sparse): {H_norm:.6e}")
print(f"Checked {len(Ugs)} U_g operators")
print(f"Number with relative commutator > {tol_rel:e}: {len(bad)}")

# show top offenders (up to 20)
print("\nTop offenders (index, ||[H,Ug]||_F, relative):")
for idx, nrm, rel in rel_values_sorted[:20]:
    tag = "!!" if rel > tol_rel else "  "
    print(f"{tag} {idx:3d}  {nrm:12.6e}  rel={rel:.3e}")

# if any noncommuting found, raise or print details
if bad:
    print("\nNon-commuting U_g indices (rel > tol):")
    for i, nrm, rel in sorted(bad, key=lambda x: x[2], reverse=True):
        print(f"  index={i:3d}, ||[H,Ug]||_F = {nrm:.6e}, rel = {rel:.3e}")
else:
    print("\nAll U_g commute with H within tolerance.")


# In[10]:


def compose(p, q): return p[q]

# find identity index
identity_idx = next(i for i, p in enumerate(perms) if np.array_equal(p, np.arange(len(p))))
identity = perms[identity_idx]

# find all involution indices (order 2, excluding identity)
involution_indices = [i for i, p in enumerate(perms) if i != identity_idx and np.array_equal(compose(p, p), identity)]

# find the unique central involution (spatial inversion)
inversion_idx = None
for i in involution_indices:
    p = perms[i]
    if all(np.array_equal(compose(p, g), compose(g, p)) for g in perms):
        inversion_idx = i
        break

print("identity index:", identity_idx)
print("involution indices:", involution_indices)
print("inversion index:", inversion_idx)
if inversion_idx is not None:
    print("inversion perm:", perms[inversion_idx].tolist())


# In[11]:


ordered_indices = sol["ordered_indices"]

# Canonical Ih column order
class_labels_Ih = [
    "E", "12C5", "12C5^2", "20C3", "15C2",
    "i", "12S10", "12(S10)^3", "20S6", "15σ",
]

# map raw class index -> canonical Ih label
raw_to_label = {raw: lbl for raw, lbl in zip(ordered_indices, class_labels_Ih)}

print("Conjugacy classes (canonical label <- raw_index):")
for k, lbl in enumerate(class_labels_Ih):
    raw_idx = ordered_indices[k]
    cls = classes[raw_idx]
    size = len(cls)
    order = perm_order(cls[0])
    print(f"  {lbl:>10}  <- raw {raw_idx:3d}    size={size:3d}, order={order:2d}")

print("\nRaw class summary (raw_idx, size, order, chi_red):")
for cd in class_data:
    label = raw_to_label.get(cd["idx"], f"raw_{cd['idx']}")
    print(f"  {label:>10}  raw_idx={cd['idx']:3d}  size={cd['size']:3d}  order={cd['order']:2d}  chi_red={cd['chi']:6.1f}")

# Build class operators in Ih order
def build_class_operators_in_Ih_order(N_spins, classes, ordered_indices):
    class_ops = []
    for class_idx in ordered_indices:
        class_perms = classes[class_idx]
        U_sum = None
        for perm in class_perms:
            U_g = build_symmetry_operator(N_spins, perm)
            if U_sum is None:
                U_sum = U_g.astype(np.complex128, copy=True).tocsr()
            else:
                U_sum = U_sum + U_g
        class_ops.append(U_sum.tocsr())
    return class_ops

def build_projectors(class_ops, chi_irreps, dims_irreps, class_sizes, G_order=120):
    """
    Build orthogonal projectors P_Gamma for each irrep Gamma of group G using sparse ops only.
    Returns a list of sparse matrices (same sparse format as used during accumulation).
    """
    projectors = []
    n_irreps, n_classes = chi_irreps.shape

    # assume at least one class_op present and use its shape
    if len(class_ops) == 0:
        return projectors
    dim = class_ops[0].shape[0]

    for i in range(n_irreps):
        d_Gamma = dims_irreps[i]
        chi_Gamma = chi_irreps[i]
        label = irrep_labels[i] if i < len(irrep_labels) else f"irrep_{i}"
        print(f"Building projector for irrep '{label}' (dim={d_Gamma})")

        # start from a sparse zero matrix (CSR) with complex dtype and accumulate sparsely
        P = csr_matrix((dim, dim), dtype=np.complex128)
        for k, C_k in enumerate(class_ops):
            class_lbl = class_labels_Ih[k] if k < len(class_labels_Ih) else f"class_{k}"
            coef = complex((d_Gamma / G_order) * np.conj(chi_Gamma[k]))
            print(f"  Adding class '{class_lbl}' contribution with chi={chi_Gamma[k]}")
            # sparse scalar multiplication + sparse addition keeps result sparse
            P = P + coef * C_k

        projectors.append(P)   # already sparse; do not convert to dense/force .tocsr()

    return projectors

# Input data (must all use the same class order!)
G_order = 120
class_sizes_Ih = np.array([1,12,12,20,15,1,12,12,20,15])
dims_irreps = np.array([1,3,3,4,5,1,3,3,4,5])

# Build classes
classop = build_class_operators_in_Ih_order(N_spins, classes, ordered_indices)

# Build projectors
projectors = build_projectors(classop, chi_irreps, dims_irreps, class_sizes_Ih, G_order)

print(f"\nBuilt {len(projectors)} projectors for irreps:", irrep_labels)

# Quick sanity checks: idempotency (P^2 = P), Hermiticity (P† = P) and trace
err_thresh = 1e-8  # threshold below which we declare the property satisfied
for i, P in enumerate(projectors):
    label = irrep_labels[i] if i < len(irrep_labels) else f"irrep_{i}"
    # P may be sparse CSR
    P2 = P @ P
    diff_idem = P2 - P
    # idempotency error (Frobenius)
    if issparse(diff_idem):
        idem_err = np.sqrt((diff_idem.multiply(diff_idem.conj())).sum())
    else:
        idem_err = np.linalg.norm(diff_idem, ord='fro')
    # Hermiticity error
    diff_herm = P - P.conj().T
    if issparse(diff_herm):
        herm_err = np.sqrt((diff_herm.multiply(diff_herm.conj())).sum())
    else:
        herm_err = np.linalg.norm(diff_herm, ord='fro')
    # trace (should equal dimension of the irrep subspace: n_Gamma * d_Gamma)
    try:
        tr = np.real(P.diagonal().sum())
    except Exception:
        tr = float(np.real(np.trace(P.toarray())))

    idem_ok = idem_err < err_thresh
    herm_ok = herm_err < err_thresh
    ok = idem_ok and herm_ok

    print(f"\nProjector '{label}':")
    print(f"  trace = {tr:.6f}")
    print(f"  Idempotent: {'YES' if idem_ok else 'NO'} (err={idem_err:.3e}, thresh={err_thresh:.1e})")
    print(f"  Hermitian : {'YES' if herm_ok else 'NO'} (err={herm_err:.3e}, thresh={err_thresh:.1e})")
    print(f"  Projector valid (idempotent & hermitian): {'YES' if ok else 'NO'}")


# In[12]:


# Check that sum of projectors = Identity
dim = projectors[0].shape[0]
Id = eye(dim, dtype=np.complex128, format='csr')

# sum projectors (sparse)
S = None
for P in projectors:
    S = P.copy() if S is None else S + P

diff = (S - Id).tocsr()

# Frobenius norm of the deviation
if issparse(diff):
    s = (diff.multiply(diff.conj())).sum()
    frob = float(np.sqrt(np.real(s)))  # take real part to avoid ComplexWarning
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz > 0 else 0.0
else:
    frob = float(np.linalg.norm(diff, ord='fro'))
    max_abs = float(np.max(np.abs(diff)))

# Total trace should equal dim
# (trace is linear, so either sum individual traces or trace of S)
total_trace = float(np.real(S.diagonal().sum()))

print(f"Sum of projector traces = {total_trace:.6f}  (expected {dim})")
print(f"|| sum(P) - I ||_F = {frob:.3e}, max_abs_entry = {max_abs:.3e}")
print("Sum is identity:" , "YES" if frob < 1e-8 and abs(total_trace - dim) < 1e-6 else "NO")


# In[13]:


eigenvalues, eigenvectors = eigsh(H, k=5, which='SA')  # smallest algebraic
print("Lowest 5 eigenvalues of H:", eigenvalues)


# In[ ]:


def classify_eigenvectors(eigvecs, projectors, irrep_labels, tol=1e-4):
    """
    Optimized version that pre-normalizes eigenvectors.
    """
    dim, nvecs = eigvecs.shape
    results = []

    for n in tqdm(range(nvecs)):
        psi = eigvecs[:, n]
        # Normalize once (eigenvectors from eigsh should already be normalized)
        psi_norm = np.sqrt(np.vdot(psi, psi).real)
        if psi_norm > 0:
            psi = psi / psi_norm

        weights = {}
        for P, label in zip(projectors, irrep_labels):
            psi_G = P @ psi  # Sparse @ dense = dense
            w = np.vdot(psi_G, psi_G).real  # No division needed since psi is normalized

            if abs(w) < tol:
                w = 0.0
            weights[label] = w
        results.append(weights)
    return results


# In[21]:


results = classify_eigenvectors(eigenvectors, projectors, irrep_labels, tol=1e-4)
for idx, weights in enumerate(results):
    # print index on its own line
    print(f"Eigenvector {idx}:")
    # print irreps decomposition on separate line (preserve canonical irrep order)
    print("  " + ", ".join(f"{lbl}: {weights[lbl]:.6f}" for lbl in irrep_labels))


# In[22]:


# tolerance for treating a weight as nonzero / significant
tol_multi = 1e-4
tol_sum = 1e-8       # tolerance for sum of weights = 1
tol_pure = 1e-6      # tolerance for testing primary weight ~= 1

# `results` is expected to be the list returned by classify_eigenvectors
# where each entry is a dict {irrep_label: weight}
nvec = len(results)
counts = Counter()
multi_members = []    # list of (vec_index, [(label, weight), ...]) for vectors with >1 significant weights

# per-vector formatted weights
per_vector_weights = []

# diagnostics
bad_sum = []
not_pure = []
pure_vectors = []

for idx, wdict in enumerate(results):
    # ensure numeric values
    wvals = np.array([float(wdict[l]) for l in irrep_labels])
    sum_w = wvals.sum()

    # assign by argmax (primary irrep)
    primary = max(wdict.items(), key=lambda kv: kv[1])[0]
    max_w = wdict[primary]
    counts[primary] += 1

    # collect significant contributions
    significant = [(lbl, wt) for lbl, wt in wdict.items() if wt > tol_multi]
    if len(significant) > 1:
        multi_members.append((idx, sorted(significant, key=lambda x: -x[1])))

    # record per-vector weights (rounded) for printing
    per_vector_weights.append((idx, {lbl: float(f"{wt:.6f}") for lbl, wt in wdict.items()}))

    # diagnostics: sum check
    if abs(sum_w - 1.0) > tol_sum:
        bad_sum.append((idx, float(sum_w)))

    # diagnostics: purity check (is the vector purely in one irrep?)
    if max_w < 1.0 - tol_pure:
        not_pure.append((idx, primary, float(max_w)))
    else:
        # record pure vectors where primary weight ≈ 1
        pure_vectors.append((idx, primary, float(max_w)))

# Print summary
print(f"Total vectors checked: {nvec}\n")
print("Counts per irrep (assigned by largest weight):")
for lbl in irrep_labels:
    print(f"  {lbl:>4}: {counts[lbl]:5d}")
print()

if multi_members:
    print(f"Vectors with >1 significant irrep contribution (tol = {tol_multi}): {len(multi_members)}")
    for idx, sig in multi_members:
        sig_str = ", ".join(f"{lbl}({wt:.6f})" for lbl, wt in sig)
        print(f"  Vector {idx}: {sig_str}")
else:
    print("No vectors have >1 significant irrep contribution (within tolerance).")

# diagnostics output
if bad_sum:
    print("\nWARNING: Vectors whose irrep-weight sums deviate from 1:")
    for idx, s in bad_sum:
        print(f"  Vec {idx:4d}: sum(weights) = {s:.8f}")
else:
    print("\nAll vectors: sum(weights) ≈ 1 (within tol_sum).")

if not_pure:
    print(f"\nVectors whose primary weight < 1 - {tol_pure} (mixed across irreps): {len(not_pure)}")
    for idx, prim, mw in not_pure:
        print(f"  Vec {idx:4d}: primary={prim}, max_weight={mw:.6f}")
else:
    print(f"\nAll vectors have primary weight ≳ 1 - {tol_pure} (appear pure): {len(pure_vectors)}")

print("\nPer-vector irrep weight coefficients (rounded to 6 decimals):")
for idx, wdict in per_vector_weights:
    wstr = ", ".join(f"{lbl}: {wdict[lbl]:.6f}" for lbl in irrep_labels)
    print(f"  Vec {idx:4d}: {wstr}")


# In[23]:


for idx in range(eigenvectors.shape[1]):
    v = eigenvectors[:, idx]
    E = eigenvalues[idx]
    print(f"\nEigenvector index: {idx}, Eigenvalue: {E:.6f}")

    # Check symmetry properties
    mag_avg = check_magnetization_sector(v, N)
    parity_sector = check_parity_sector(v, N)

    print(f"  Magnetization: {mag_avg:.6f}")
    print(f"  Parity: {parity_sector}")


# In[24]:


idx_Ug = 71 # 0 if you want to apply identity
Ug = Ugs[idx_Ug]          # sparse CSR permutation operator
Ug_matvec = lambda v: Ug @ v

print(f"Applying Ug index {idx_Ug} to the 6 Lanczos eigenvectors\n")

for si in range(eigenvectors.shape[1]):
    v = eigenvectors[:, si]
    w = Ug_matvec(v)
    lam = np.vdot(v, w) / np.vdot(v, v)        # Rayleigh estimate of eigenvalue
    resid = np.linalg.norm(w - lam * v)       # how close w == lam*v
    sign_check = None
    if np.allclose(lam, 1.0, atol=1e-6):
        sign_check = "+1"
    elif np.allclose(lam, -1.0, atol=1e-6):
        sign_check = "-1"
    else:
        sign_check = "not ±1"

    # avoid formatting a string with float format spec -> build imag part separately
    imag_part = "" if abs(lam.imag) < 1e-12 else f"{lam.imag:+.6f}j"
    print(f"vec {si:5d}: lambda = {lam.real:+.6f}{imag_part}, resid = {resid:.2e} -> {sign_check}")


# In[41]:


# --- CSR helpers for sparse column vectors (shape (D,1)) ---
def _csr_rand_vec_in_indices(D, idx_sector, rng):
    rows = np.asarray(idx_sector, dtype=np.int64)
    K = rows.size
    cols = np.zeros(K, dtype=np.int64)
    data = rng.normal(size=K) + 1j * rng.normal(size=K)
    return csr_matrix((data, (rows, cols)), shape=(D, 1), dtype=np.complex128)

def _csr_basis_vec(D, b):
    return csr_matrix(([1.0 + 0.0j], ([int(b)], [0])), shape=(D, 1), dtype=np.complex128)

def _csr_norm(v):
    if v.nnz == 0:
        return 0.0
    return float(np.sqrt(np.sum(np.abs(v.data) ** 2)))

def _csr_normalize(v):
    nrm = _csr_norm(v)
    if nrm > 0:
        v = v.copy()
        v.data /= nrm
    return v, nrm

def _csr_residual(Ug, v):
    r = Ug @ v - v
    return _csr_norm(r)

def _csr_parity_label(v, N, tol=1e-6):
    # compute weights on even/odd parity using nonzeros only
    coo = v.tocoo()
    rows = coo.row
    data = coo.data
    w_even = 0.0
    w_odd = 0.0
    for idx, amp in zip(rows, data):
        if parity(int(idx), N) == 1:
            w_even += float((amp.conjugate() * amp).real)
        else:
            w_odd += float((amp.conjugate() * amp).real)
    if w_even > 1.0 - tol:
        return "even"
    if w_odd > 1.0 - tol:
        return "odd"
    return f"mixed (even={w_even:.4f}, odd={w_odd:.4f})"

def _csr_magnetization_avg(v, N):
    coo = v.tocoo()
    rows = coo.row
    data = coo.data
    m = 0.0
    for idx, amp in zip(rows, data):
        m += magnetization(int(idx), N) * float((amp.conjugate() * amp).real)
    return m

def make_vc0_parity_Ih_irrep(N, projectors, par, irrep_labels, target_irrep,
                                 rng=None, max_attempts=200):
    """
    Build a single normalized vector v with:
      - even/odd parity,
      - in the chosen Ih irrep subspace (e.g. "Ag" or "Gg").
    Sparse (CSR) internally. Returns dense 1D vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    D = 1 << N
    if par == "even":
        idx_sector = [b for b in range(D) if parity(b, N) == 1]
    else:
        idx_sector = [b for b in range(D) if parity(b, N) == -1]
    if len(idx_sector) == 0:
        raise RuntimeError(f"No {par}-parity basis states")

    try:
        i_target = irrep_labels.index(str(target_irrep))
    except ValueError:
        raise ValueError(f"target_irrep '{target_irrep}' not found in irrep_labels")
    P_target = projectors[i_target].astype(np.complex128, copy=False).tocsr()

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        v = _csr_rand_vec_in_indices(D, idx_sector, rng)

        v = P_target @ v
        v, nrm = _csr_normalize(v)
        if nrm > 0:
            w = _csr_norm(P_target @ v) ** 2
            par = _csr_parity_label(v, N)
            if w > 1e-8:
                return v.toarray().ravel()
            return v.toarray().ravel()

    # fallback
    for b in idx_sector:
        v = _csr_basis_vec(D, b)
        v = P_target @ v
        v, nrm = _csr_normalize(v)
        if nrm > 0:
            return v.toarray().ravel()

    raise RuntimeError(f"Failed to build vector in irrep '{target_irrep}' after {max_attempts} attempts")


# In[ ]:


# build a vector in a chosen Ih irrep (no Ug invariance enforced)
vc0 = make_vc0_parity_Ih_irrep(N, projectors, par="odd", irrep_labels=irrep_labels, target_irrep="Gg", rng=None)

# quick checks for vc0
tol = 1e-8

print("norm:", np.vdot(vc0, vc0).real)

mag = check_magnetization_sector(vc0, N)
pari = check_parity_sector(vc0, N)
print("magnetization (expected 0):", mag)
print("parity sector (expected 'odd'):", pari)

support = np.nonzero(np.abs(vc0) > 1e-10)[0]
print("support size:", support.size)

# compute membership weights in each Ih irrep: w_Gamma = || P_Gamma |v> ||^2
weights = {}
for P, lbl in zip(projectors, irrep_labels):
    psi_G = P @ vc0
    w = np.vdot(psi_G, psi_G).real
    weights[lbl] = float(w)

# summary: sorted by weight
sorted_weights = sorted(weights.items(), key=lambda kv: -kv[1])
print("\nIrrep weights (descending):")
for lbl, w in sorted_weights:
    print(f"  {lbl:>4}: {w:.6e}")

total_weight = sum(weights.values())
print(f"\nSum of irrep-weights = {total_weight:.12f} (should be ≈ 1.0)")

# optional: overlap with Ug (diagnostic only, not required)
if 'Ug' in globals():
    resid = np.linalg.norm(Ug @ vc0 - vc0)
    ov = (np.vdot(vc0, Ug @ vc0) / np.vdot(vc0, vc0)).real
    print(f"\nDiagnostic Ug overlap: ||Ug v - v|| = {resid:.2e}, <v|Ug|v> = {ov:.6f}")

# sanity checks (raise only if major failures)
if pari.startswith("odd"):
    print("Vector is in odd-parity sector as expected.")
else:
    print("Vector is in even-parity sector as expected.")

if abs(total_weight - 1.0) > 1e-6:
    print("WARNING: sum of irrep weights deviates from 1 by", total_weight - 1.0)


# In[44]:


# run many short Chebyshev filters (m=500) repeatedly instead of one huge m
m = 1000
n_steps = 50
pad = 0.05
use_jackson = True

start_vec = vc0
current = start_vec
Emin = -62.51489576
Emax = 62.60812227
target = eigenvalues[4]  # 5th lowest eigenvalue from earlier Lanczos run
print(f"Target eigenvalue for filtering: {target:.8f}")
print(f"Target support: {np.count_nonzero(np.abs(eigenvectors[:,4]) > 1e-10)}")

print(f"Running {n_steps} iterations with m={m}, pad={pad}")

last_eval = None
try:
    for i in range(1, n_steps + 1):
        Phi, evals = chebyshev_filter_v0_numpy(H, current, Emin=Emin, Emax=Emax,
                                               target_E0=target, m=m, pad=pad, use_jackson=use_jackson)
        # basic sanity checks
        if not np.isfinite(evals):
            print(f"Stopped at step {i}: non-finite eval {evals}")
            break
        if not np.all(np.isfinite(Phi)):
            print(f"Stopped at step {i}: non-finite entries in Phi")
            break

        current = Phi
        last_eval = evals

        # print every ~10% of total steps (works for n_steps=10 too)
        if (i % max(1, n_steps // 10) ) == 0 or i == 1 or i == n_steps:
            support = np.count_nonzero(np.abs(current) > 1e-10)
            print(f"step {i:4d}: approx E = {last_eval:.8f}, support = {support}")

except RuntimeError as e:
    print("RuntimeError during filtering:", e)

# final result in `current`, last Rayleigh estimate in `last_eval`
Phi_final = current
E_final = last_eval
print("Done. Final approx E:", E_final)
print("Final support size:", np.count_nonzero(np.abs(Phi_final) > 1e-10))


# In[ ]:


# run many short Chebyshev filters (m=1000) repeatedly instead of one huge m
m = 1000
n_steps = 50
pad = 0.05
use_jackson = True

start_vec = vc0
current = start_vec
Emin = -62.51489576
Emax = 62.60812227
target = -18.0

print(f"Running {n_steps} iterations with m={m}, pad={pad}")

last_eval = None
try:
    for i in range(1, n_steps + 1):
        Phi, evals = chebyshev_filter_v0_numpy(H, current, Emin=Emin, Emax=Emax,
                                               target_E0=target, m=m, pad=pad, use_jackson=use_jackson)
        # basic sanity checks
        if not np.isfinite(evals):
            print(f"Stopped at step {i}: non-finite eval {evals}")
            break
        if not np.all(np.isfinite(Phi)):
            print(f"Stopped at step {i}: non-finite entries in Phi")
            break

        current = Phi
        last_eval = evals

        # print every ~10% of total steps (works for n_steps=10 too)
        if (i % max(1, n_steps // 10) ) == 0 or i == 1 or i == n_steps:
            support = np.count_nonzero(np.abs(current) > 1e-10)
            print(f"step {i:4d}: approx E = {last_eval:.8f}, support = {support}")

except RuntimeError as e:
    print("RuntimeError during filtering:", e)

# final result in `current`, last Rayleigh estimate in `last_eval`
Phi_final = current
E_final = last_eval
print("Done. Final approx E:", E_final)
print("Final support size:", np.count_nonzero(np.abs(Phi_final) > 1e-10))

