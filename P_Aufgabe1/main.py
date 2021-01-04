import numpy as np

from lib import timedcall, plot_2d


def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate product of two matrices a * b.

    Arguments:
    a : first matrix
    b : second matrix

    Return:
    c : matrix product a * b

    Raised Exceptions:
    ValueError: if matrix sizes are incompatible

    Side Effects:
    -

    Forbidden: numpy.dot, numpy.matrix
    """

    n, m_a = a.shape
    m_b, p = b.shape

    # TODO: test if shape of matrices is compatible and raise error if not
    if m_b != m_a:
        raise ValueError("shape of matrices is not compatible")
    # Initialize result matrix with zeros
    c = np.zeros((n, p))

    # TODO: Compute matrix product without the usage of numpy.dot()
    for i in range(len(a)):
        for j in range(len(b[0])):
            for h in range(len(b)):
                c[i][j] += a[i][h] * b[h][j]


    return c


def compare_multiplication(nmax: int, n: int) -> dict:
    """
    Compare performance of numpy matrix multiplication (np.dot()) and matrix_multiplication.

    Arguments:
    nmax : maximum matrix size to be tested
    n : step size for matrix sizes

    Return:
    tr_dict : numpy and matrix_multiplication timings and results {"timing_numpy": [numpy_timings],
    "timing_mat_mult": [mat_mult_timings], "results_numpy": [numpy_results], "results_mat_mult": [mat_mult_results]}

    Raised Exceptions:
    -

    Side effects:
    Generates performance plots.
    """

    " *** in which RANGE should i create a random matrix "
    rang_of_random_numbers = 10

    x, y_mat_mult, y_numpy, r_mat_mult, r_numpy = [], [], [], [], []
    tr_dict = dict(timing_numpy=y_numpy, timing_mat_mult=y_mat_mult, results_numpy=r_numpy, results_mat_mult=r_mat_mult)

    for m in range(2, nmax, n):

        # TODO: Create random mxm matrices a and b
        a = np.random.randint(rang_of_random_numbers, size=(n, n))
        b = np.random.randint(rang_of_random_numbers, size=(n, n))

        # Execute functions and measure the execution time
        time_mat_mult, result_mat_mult = timedcall(matrix_multiplication, a, b)
        time_numpy, result_numpy = timedcall(np.dot, a, b)

        # Add calculated values to lists
        x.append(m)
        y_numpy.append(time_numpy)
        y_mat_mult.append(time_mat_mult)
        r_numpy.append(result_numpy)
        r_mat_mult.append(result_mat_mult)

    # Plot the computed data
    plot_2d(x_data=x, y_data=[y_mat_mult, y_numpy], labels=["matrix_mult", "numpy"],
            title="NumPy vs. for-loop matrix multiplication",
            x_axis="Matrix size", y_axis="Time", x_range=[2, nmax])

    return tr_dict


def machine_epsilon(fp_format: np.dtype) -> np.number:
    """
    Calculate the machine precision for the given floating point type.

    Arguments:
        fp_format (object):
    fp_format: floating point format, e.g. float32 or float64

    Return:
    eps : calculated machine precision

    Raised Exceptions:
    -

    Side Effects:
    Prints out iteration values.

    Forbidden: numpy.finfo
    """

    # TODO: create epsilon element with correct initial value and data format fp_format
    one = 0
    tow = 0
    eps = 0
    type = fp_format
    if type is np.float64:
        eps = np.float64(1.0)
    if type is np.complex128:
        eps = np.complex128(1.0)
    if type is np.float32:
        eps = np.float32(1.0)

    # Create necessary variables for iteration
    if type is np.float64:
        one = np.float64(1.0)
        two = np.float64(2.0)
    if type is np.float32:
        one = np.float32(1.0)
        two = np.float32(2.0)
    if type is np.complex128:
        one = np.complex128(1.0)
        two = np.complex128(2.0)
    i = 0

    print('  i  |       2^(-i)        |  1 + 2^(-i)  ')
    print('  ----------------------------------------')

    # TODO: determine machine precision without the use of numpy.finfo()
    while one + eps != one:
        eps = eps / two
        i += 1
    i -= 1
    eps *= 2
    print('{0:4.0f} |  {1:16.8e}   | equal 1'.format(i, eps))

    return eps


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create 2x2 rotation matrix around angle theta.

    Arguments:
    theta : rotation angle (in degrees)

    Return:
    r : rotation matrix

    Raised Exceptions:
    -

    Side Effects:
    -
    """

    # create empty matrix
    r = np.zeros((2, 2))

    # TODO: convert angle to radians
    ''' also 180 Degrees is equal to 3.1415 Radians '''
    radian = np.radians(theta)


    # TODO: calculate diagonal terms of matrix
    r[0][0] = np.cos(radian)
    r[1][1] = np.cos(radian)


    # TODO: off-diagonal terms of matrix
    r[0][1] = np.sin(radian) * (-1)
    r[1][0] = np.sin(radian)

    return r


def inverse_rotation(theta: float) -> np.ndarray:
    """
    Compute inverse of the 2d rotation matrix that rotates a
    given vector by theta.

    Arguments:
    theta: rotation angle

    Return:
    Inverse of the rotation matrix

    Forbidden: numpy.linalg.inv, numpy.linalg.solve
    """

    # TODO: compute inverse rotation matrix
    r = np.zeros((2, 2))
    radian = np.radians(theta)
    for i in range(2):
        r[i][i] = np.cos(radian)
    r[0][1] = np.sin(radian) * (-1)
    r[1][0] = np.sin(radian)
    m = np.zeros((2, 2))
    ad_bc = (r[0][0]*r[1][1])-(r[0][1]*r[1][0])
    # print(type(ad_bc))
    det = 1.0 / ad_bc
    # print(type(det))
    for i in range(2):
        m[i][i] = np.cos(radian) * det
    m[0][1] = np.sin(radian) * det
    m[1][0] = np.sin(radian) * (-1) * det


    return m


if __name__ == '__main__':

    # print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
    #       "server for the grading.\nTo test your implemented functions you can "
    #       "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    # a = np.array([[1, 2, 3], [3, 4, 5], [7, 8, 4]])
    # b = np.array([[1, 2, 3], [3, 4, 5], [7, 8, 4]])
    # compare_multiplication(10, 4)
    # print("\n-------------------------------------\n")
    machine_epsilon(np.float32)
    print("\n-------------------------------------\n")
    print(np.finfo(np.float32))
    # print("\n-------------------------------------\n")
    # print(rotation_matrix(90))
    # print("\n-------------------------------------\n")
    # print(inverse_rotation(90))