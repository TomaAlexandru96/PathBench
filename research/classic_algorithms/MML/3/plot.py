from answers import *
from answersCW2 import *
import matplotlib.pyplot as plt


def get_args(fs, N=25):
    X, Y = get_db(N, (0, 0.9))

    X = np.array(X)
    Y = np.array(Y)

    design_matrix = np.array(poly_mat(fs, X))
    return design_matrix, X, Y


def gradient_descent_wrapper(iterations, start_point, step_size, design_matrix, Y, extents, with_plot):
    def lml_wrapper(alpha, beta):
        return lml(alpha, beta, design_matrix, Y)

    def grad_lml_wrapper(alpha, beta):
        return grad_lml(alpha, beta, design_matrix, Y)

    return gradient_descent(iterations, lml_wrapper, grad_lml_wrapper, start_point, step_size, extents, with_plot)


def gradient_descent(iterations, fun, gradient, starting_point, step_size, extents, with_plot):
    x = starting_point
    xs = [starting_point]

    for i in range(iterations):
        x = x + step_size * gradient(x[0], x[1])
        xs.append(x)

    if with_plot:
        X = np.linspace(extents[0], extents[1])
        Y = np.linspace(extents[2], extents[3])
        X, Y = np.meshgrid(X, Y)

        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i][j] = fun(X[i][j], Y[i][j])
                # print(str(Z[i][j]) + ": " + str(X[i][j]) + " " + str(Y[i][j]))

        plot(starting_point, X, Y, Z, xs, step_size)
    return xs[-1]


def plot(start, X, Y, Z, xs, step_size):
    # plt.imshow(Z, extent=extents, origin='lower', cmap='Greens', alpha=0.8)
    scatter = plt.scatter(list(map(lambda a: a[0], xs)), list(map(lambda a: a[1], xs)), c=range(len(xs)), s=20, cmap="summer")
    contours = plt.contour(X, Y, Z, levels=30, cmap="rainbow")
    plt.title("Step size: " + str(step_size) + ", Starting point: " + str(start) + ", End point: " + str(xs[-1]))
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    sbar = plt.colorbar(scatter)
    cbar = plt.colorbar()
    sbar.ax.set_ylabel("Iterations")
    cbar.ax.set_ylabel("Function value")
    plt.clabel(contours, inline=True, fontsize=8)
    plt.show()


def f_b():
    iterations = 150
    start_point = np.array([0.5, 0.65])
    step_size = 0.01
    extents = list(map(lambda a: a * 1, [0.3, 0.7, 0.3, 0.7]))
    fs = get_fs_poly(1)
    design_matrix, X, Y = get_args(fs)
    gradient_descent_wrapper(iterations, start_point, step_size, design_matrix, Y, extents, True)


def f_c():
    iterations = 10000
    start_point = np.array([1.5, 2])
    step_size = 0.01
    extents = list(map(lambda a: a * 1, [0.1, 4, 0.1, 4]))
    degrees = 11

    vals = []

    for i in range(degrees + 1):
        fs = get_fs_trigo(i)
        design_matrix, X, Y = get_args(fs)
        x = gradient_descent_wrapper(iterations, start_point, step_size, design_matrix, Y, extents, False)
        vals.append(lml(x[0], x[1], design_matrix, Y))

    plt.plot(list(range(degrees + 1)), vals)
    plt.xticks(range(degrees + 1))
    plt.xlabel("Degree")
    plt.ylabel("Function Value")
    plt.show()


def f_d():
    data_points = 25
    alpha = 1
    beta = 0.1
    degrees = 10
    samples_size = 5
    colours = ["red", "blue", "green", "cyan", "brown"]
    fs = get_fs_gauss(degrees, 0.1, (-0.5, 1))
    design_matrix, X, Y = get_args(fs, data_points)

    k = design_matrix.T.shape[0]
    S0 = alpha * np.identity(k)
    S0_inv = np.linalg.inv(S0)
    Sn = np.linalg.inv(S0_inv + beta ** (-1) * np.dot(design_matrix.T, design_matrix))
    mn = np.dot(Sn, beta ** (-1) * np.dot(design_matrix.T, Y))

    samples = [np.random.multivariate_normal(mn, Sn) for _ in range(samples_size)]

    X_test, Y_test = get_db(200, (-1, 1.5))
    design_matrix_new = np.array(poly_mat(fs, X_test))
    predictive_mean = np.zeros(len(X_test))

    labels = []
    # print data
    l = plt.scatter(X_test, Y_test, label=str("Data"), s=12)
    labels.append(l)

    for i, sample in enumerate(samples):
        Y_val = np.dot(design_matrix_new, sample)
        l, = plt.plot(X_test, Y_val, color=colours[i], label=str("Sample with weights: " + str((sample * 100).astype(int) / 100)))
        labels.append(l)
        predictive_mean += Y_val

    predictive_mean *= 1.0 / samples_size
    l, = plt.plot(X_test, predictive_mean, color="gold", label=str("Predictive mean"))
    labels.append(l)

    standard_deviation = list(map(lambda a: np.sqrt(a), np.diag(np.dot(np.dot(design_matrix_new, Sn), design_matrix_new.T))))

    # noise error
    Y_Error_Plus = predictive_mean + standard_deviation
    Y_Error_Minus = predictive_mean - standard_deviation

    Y_ErrorBar_Plus = predictive_mean + (standard_deviation + np.full(predictive_mean.shape, beta))
    Y_ErrorBar_Minus = predictive_mean - (standard_deviation + np.full(predictive_mean.shape, beta))

    l, = plt.plot(X_test, Y_ErrorBar_Plus, color="black", linestyle=":",
                  label=str("Error bar (including noise)"))
    labels.append(l)
    plt.plot(X_test, Y_ErrorBar_Minus, color="black", linestyle=":")

    # shaded error
    plt.fill_between(X_test, Y_Error_Minus, Y_Error_Plus, color="grey", alpha=0.2)

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.legend(prop={'size': 7}, handles=labels)
    plt.show()


if __name__ == '__main__':
    f_d()
