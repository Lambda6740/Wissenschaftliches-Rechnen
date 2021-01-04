import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination
def fs(A, b):
    kk = np.zeros(A.shape[1])
    for h in range(0, A.shape[0]):
        #if np.isclose(A[h, h], 0):
        #    raise ZeroDivisionError
        b[h] = b[h] / A[h, h]
        A[h, h] = 1
        kk[h] = b[h]
        for h in range(h + 1, A.shape[0]):
            b[h] = b[h] - A[h, h] * kk[h]
            A[h, h] = 0

    return kk


def swap_rows(A, b, i):
    n = b.size
    if np.abs(A[i, i]) == 0:
        for k in range(i + 1, n):
            if np.abs(A[k, i]) > np.abs(A[i, i]):
                A[[i, k]] = A[[k, i]]
                b[[i, k]] = b[[k, i]]
                break
    return A, b

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A = A.copy()
    b = b.copy()
    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    m, q = np.shape(A)
    if m != q:
        raise ValueError("Only square matrices are allow.")
    v = b.size
    if m != v:
        raise ValueError("shape")
    # TODO: Perform gaussian elimination
    k = q+1
    new_A = np.ndarray((m, k), dtype=float)
    u, o = new_A.shape
    for i in range(0, u-1):
        for j in range(0, o-1):
            if j == o-1:
                new_A[i, j] = b[i, 0]
            new_A[i, j] = A[i, j]
    print(b)
    print(new_A)
    n = A.shape[0]
    for row in range(0, n - 1):
        if use_pivoting:
            pass
        for h in range(row + 1, n):
            fac = A[h, row] / A[row, row]
            for k in range(row, n):
                A[h, k] = A[h, k] - fac * A[row, k]
            b[h] = b[h] - fac * b[row]
    # print("A = \n%s \nand b = \n%s" % (A, b))
    print(type(A))
    print(type(b))
    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    m_n, n_m = np.shape(A)
    n = len(b)
    if m_n != n:
        raise ValueError("matrix and vector is compatible")

    for k in range(0, n):
        if A[k, k] == 0:
            raise ValueError
    # TODO: Initialize solution vector with proper size
    x = np.zeros_like(b)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    n = len(A)
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            b[i] = b[i] - A[i, j]*x[j]
        x[i] = b[i]/A[i, i]

    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if not np.allclose(M, np.transpose(M)):
        raise ValueError("Matrix is not symmetric!")

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    cf = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                add = 0
                for k in range(0, i):
                    add = add + cf[i, k] ** 2
                if M[i, i] - add <= 0:
                    raise ValueError("Matrix shoud be positive semi-definite!")
                cf[i, j] = np.sqrt(M[i, i] - add)
            elif i > j:
                add = 0
                for k in range(0, j):
                    add = add + (cf[i, k] * cf[j, k])
                cf[i, j] = (M[i, j] - add) / cf[j, j]
    return cf


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if n != m:
        raise ValueError("L soll quadratisch sein!")

    # TODO Solve the system by forward- and backsubstitution
    for i in range(0, n):
        for j in range(0, i):
            if not np.isclose(L[j, i], 0):
                raise ValueError("L ist nicht untere dreiecksmatrix ")
    if n != b.shape[0]:
        raise ValueError("shape of matrices is not compatible")

    y = fs(L, b)
    print(y, "\n")
    a_t = np.transpose(L)
    x = back_substitution(a_t, y)
    print(x, "\n")

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_shots * n_rays, n_grid * n_grid))

    # TODO: Initialize intensity vector
    g = np.zeros((n_rays, n_shots))

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    for j in range(0, n_shots):
        theta = np.pi * (j / n_shots)
        # Take a measurement with the tomograph from direction r_theta.
        # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
        # ray_indices: indices of rays that intersect a cell
        # isect_indices: indices of intersected cells
        # lengths: lengths of segments in intersected cells
        # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        for i in range(0, n_rays):
            g[i, j] = intensities[i]
        for k in range(0, len(lengths)):
            L[j * n_rays + ray_indices[k], isect_indices[k]] = lengths[k]

    o = np.zeros(n_shots * n_rays)
    for i in range(0, n_shots):
        for j in range(0, n_rays):
            o[i * n_rays + j] = g[j, i]
    return L, o


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)
    densities = np.linalg.solve(np.dot(np.transpose(L), L), np.dot(np.transpose(L), g))
    # TODO: Convert solution of linear system to 2D image
    cnj = np.zeros((n_grid, n_grid))
    for i in range(0, n_grid):
        for j in range(0, n_grid):
            cnj[i, j] = densities[(n_grid * i) + j]

    return cnj


def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")

    #A = symmetrize(np.random.randn(2, 2))
    #b = np.array([[6.0], [-4.0], [27.0]])
    A = np.random.randn(4, 4)
    x = np.random.rand(4)
    b = np.dot(A, x)
    # A = np.array([[5, 1, 0, 2, 1], [0, 4, 0, 1, 2], [1, 1, 4, 1, 1], [0, 1, 2, 6, 0], [0, 0, 1, 2, 4]])
    # b = np.array([[1], [2], [3], [4], [5]])
    L, b = gaussian_elimination(A, b, True)
    print(L)
    print(b)
    b = np.linalg.solve(A, b)
    print(b)
    # A_t = np.invert(A)
    # solve_cholesky(A, b)
