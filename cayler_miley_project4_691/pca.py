import numpy as np


def compute_Z(X, centering=True, scaling=False):
    Z = X

    if centering:
        Z = Z - Z.mean(axis=0)

    if scaling:
        Z = Z/Z.std(axis=0)

    return Z


def compute_covariance_matrix(Z):
    return Z.T @ Z


def find_pcs(COV):
    eigenvalues, eigenvectors = np.linalg.eig(COV)

    # sort vectors by eigenvalues
    indices = np.argsort(eigenvalues)

    eigenvalues = np.flip(eigenvalues[indices], axis=0)
    eigenvectors = np.flip(eigenvectors[indices], axis=0)

    return eigenvalues, eigenvectors.T


def project_data(Z, PCS, L, k, var):
    if k > Z[0].size:
        return Z @ PCS

    if var > 0:
        L_var = L.cumsum()/L.sum()  # cumulative variance
        k = np.max(np.argwhere(var >= L_var)) + 1

    return Z @ PCS[:, :k]
