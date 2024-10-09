import numpy as np
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
from sklearn.cluster import KMeans

# Helper function
def Afun(A, C, b, sigma):
    y1 = np.linalg.solve(C.T @ C, C.T @ b)
    y2 = b - C @ y1
    y3 = A @ y2
    y4 = np.linalg.solve(C.T @ C, C.T @ y3)

    return y3 - C @ y4 - sigma * y2 + sigma * b


# s-Fair-SC (code as provided in MATLAB in original paper repo, translated to python)
def s_fair_sc(W, D, F, k):
    # INPUT:
    #   W ... (weighted) adjacency matrix of size n x n
    #   D ... degree matrix of W
    #   F ... group membership matrix G of size n x (h-1)
    #   k ... number of clusters
    #
    # OUTPUT:
    # clusterLabels ... vector of length n comprising the cluster label for each data point
    
    n = W.shape[0]

    # Calculate Laplacian
    L = D - W

    sqrtD = sqrtm(D)
    C = np.linalg.solve(sqrtD, F)  # Normalize F by sqrtD
    Ln = np.linalg.solve(sqrtD, L) @ np.linalg.inv(sqrtD)  # Normalized Laplacian
    Ln = (Ln + Ln.T) / 2  # Ensure symmetry
    sigma = np.linalg.norm(Ln, 1)  # 1-norm of the normalized Laplacian

    def Afun_mv(b):
        if b.ndim<2:
            b = b.reshape((b.size,1))
        return Afun(Ln,C,b,sigma)

    A = LinearOperator((n,n),matvec=Afun_mv)
    
    # Calculate k eigenvalues
    _vals, X = eigs(A, k=k, which='SR', maxiter=1000)
    
    # Normalize the eigenvectors with respect to sqrtD
    H = np.linalg.solve(sqrtD, X)
    
    # Use k-means to cluster the rows of H
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=500)
    cluster_labels = kmeans.fit_predict(H.real)
    
    return cluster_labels.tolist()