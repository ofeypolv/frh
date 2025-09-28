import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh

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


def transverse_field_ising_dodecahedral(N, J, h):
    """
    Constructs the Hamiltonian for the transverse field Ising model on a dodecahedral molecular structure.

    Parameters:
        N (int): Number of spins (should match the dodecahedral molecule, typically N=12).
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
    
    Returns:
        H (scipy.sparse.csr_matrix): The Hamiltonian matrix in sparse format.
    """
    if N != 20:
        raise ValueError("Dodecahedral molecules typically have N = 20 sites.")

    # Sparse identity matrix
    I = identity(2, format="csr")
    
    # Pauli matrices as sparse matrices
    X = csr_matrix(pauli_x())
    
    # Initialize the Hamiltonian
    H = csr_matrix((2**N, 2**N), dtype=np.float64)

    # Get Dodecahedron bonds
    bonds = dodecahedral_bonds()

    # Interaction term: J * sigma_i^x * sigma_j^x for Dodecahedron connectivity
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
        N (int): Number of spins (should match the icosahedral molecule, typically N=20).
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

def H_shifted(H0, x):
    H_s = H0 - x * identity(H0.shape[0], format="csr")
    H_s = H_s.tocsr()
    return H_s


def H_shifted_sq(H0, x):
    H_s = H0 - x * identity(H0.shape[0], format="csr")
    H_s_sq = H_s @ H_s
    H_s_sq = H_s_sq.tocsr()
    return H_s_sq

if __name__ == "__main__":
    N = 20
    J = 1.0
    h = 3.0

    H = transverse_field_ising_dodecahedral(N, J, h)
    # Hi = ising_dodecahedron(N, J)  # Unused
    # Htf = transverse_field_dodecahedral(N, h)  # Unused

    E0 = -18.0
    H_shifted = H - E0 * identity(H.shape[0], format="csr")
    #H_sq = H_shifted @ H_shifted # with this which = 'SA' finds the smallest eigenvalue of H^2, which is (E0 - E)^2
    eivt, eigt = eigsh(H_shifted, k=5, which='SM', maxiter=500, ncv=128, tol=1e-8)
    #If H_sq: The closest eigenvalue of H is: E0 ± sqrt(eivt[0])
    print(eivt + E0)