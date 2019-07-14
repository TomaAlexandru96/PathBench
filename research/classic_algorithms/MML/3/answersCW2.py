import numpy as np
import matplotlib.pyplot as plt


def poly_mat(fs, xs):
    mat = []
    for x in xs:
        row = []
        for f in fs:
            row.append(f(x))
        mat.append(row)
    return mat


def get_weights(mat, y):
    mat_t_mat = np.matmul(np.transpose(mat), mat)
    inv_mat_t_mat = np.linalg.inv(mat_t_mat)
    mat_part = np.matmul(inv_mat_t_mat, np.transpose(mat))
    return np.dot(mat_part, y)


def get_db(N, range):
    X = np.reshape(np.linspace(range[0], range[1], N), (N, 1))
    Y = np.cos(10 * X ** 2) + 0.1 * np.sin(100 * X)
    X = list(map(lambda x: x[0], X))
    Y = list(map(lambda y: y[0], Y))
    return X, Y


def mle(weights, fs, xs):
    ffs = list(map(lambda x: [f(x) for f in fs], xs))
    ys = []
    for ff in ffs:
        ys.append(np.matmul(np.transpose(weights), ff))
    return ys


def make_f_poly(degree):
    def f(x):
        return x ** degree
    return f


def get_fs_poly(degree):
    return [make_f_poly(i) for i in range(degree + 1)]


def make_f_trigo_sin(j):
    def f(x):
        return np.sin(2 * np.pi * j * x)
    return f


def make_f_trigo_cos(j):
    def f(x):
        return np.cos(2 * np.pi * j * x)
    return f


def make_f_gauss(l, u):
    def f(x):
        return np.exp(-((x - u) ** 2) / (2 * l ** 2))
    return f


def get_fs_trigo(degree):
    fs = [make_f_poly(0)]
    for i in range(1, degree + 1):
        fs.append(make_f_trigo_sin(i))
        fs.append(make_f_trigo_cos(i))
    return fs


def get_fs_gauss(degrees, l, range_us):
    fs = [make_f_poly(0)]
    us = np.linspace(range_us[0], range_us[1], degrees)
    for u in us:
        fs.append(make_f_gauss(l, u))
    return fs


def plot(get_fs, degrees, colours, range, ylim):
    N = 25
    X, Y = get_db(N, (0, 0.9))
    weights = []
    names = []
    fs = []
    for degree in degrees:
        fs.append(get_fs(degree))
        weights.append(get_weights(poly_mat(fs[-1], X), Y))
        names.append("degree " + str(degree))

    N = 200
    X, Y = get_db(N, range)
    l = plt.scatter(X, Y, s=5, label="g(x)")
    legends = [l]
    for idx, colour in enumerate(colours):
        line, = plt.plot(X, mle(weights[idx], fs[idx], X), color=colour, label=names[idx])
        legends.append(line)

    plt.legend(handles=legends)
    plt.ylim(top=ylim[1])
    plt.ylim(bottom=ylim[0])
    plt.show()


def cross_validation(folds, fs):
    X, Y = get_db(folds, (0, 0.9))
    error = 0
    for i in range(folds):
        testX = [X[i]]
        testY = [Y[i]]
        trainingX = X[:i] + X[i+1:]
        trainingY = Y[:i] + Y[i+1:]

        # train
        weights = get_weights(poly_mat(fs, trainingX), trainingY)

        # test
        actual = mle(weights, fs, testX)
        error = error + (testY[0] - actual) ** 2

    return float(error) / float(folds)


def get_mle_variation(folds, fs):
    X, Y = get_db(folds, (0, 0.9))
    weights = get_weights(poly_mat(fs, X), Y)
    variation = 0
    for i in range(folds):
        variation = variation + (Y[i] - mle(weights, fs, [X[i]])) ** 2
    return float(variation) / float(folds)


def f1a():
    plot(get_fs_poly, [0, 1, 2, 3, 11], ["red", "blue", "pink", "cyan", "green"], (-0.3, 1.3), (-4, 7))


def f1b():
    # degrees = list(map(lambda x: 2 * x + 1, [1, 11]))
    degrees = [1, 11]
    plot(get_fs_trigo, degrees, ["red", "blue"], (-1, 1.2), (-2, 2))


def f1c():
    folds = 25
    degrees = 11
    cross = []
    var = []
    for degree in range(degrees):
        fs = get_fs_trigo(degree)
        cross.append(cross_validation(folds, fs))
        var.append(get_mle_variation(folds, fs))
    squared_legend, = plt.plot(range(degrees), cross, label="Test error")
    var_legend, = plt.plot(range(degrees), var, label="Standard error", color="red")
    plt.legend(handles=[squared_legend, var_legend])
    plt.xticks(range(degrees))
    plt.ylim(top=0.6)
    plt.ylim(bottom=-0.05)
    plt.show()


def ridge_regularisation(degree, mat, Y, learning_rate):
    sigma2 = 1.0
    b2 = 1.0 / float(2 * learning_rate)
    mat_t_mat = np.matmul(np.transpose(mat), mat)
    mat_i = mat_t_mat + (sigma2 / b2) * np.identity(degree + 1)
    inv_mat_t_mat = np.linalg.inv(mat_i)
    mat_part = np.matmul(inv_mat_t_mat, np.transpose(mat))
    return np.dot(mat_part, Y)


def f2b():
    degree = 20
    learning_rate = [0.001, 20, 0.0000001]
    X, Y = get_db(25, (0, 0.9))
    fs = get_fs_gauss(degree, 0.1, (0, 1))
    weights = []

    for rate in learning_rate:
        weights.append(ridge_regularisation(degree, poly_mat(fs, X), Y, rate))

    X, Y = get_db(200, (-0.3, 1.3))
    l = plt.scatter(X, Y, s=5, label="g(x)")
    ls = [l]

    legend, = plt.plot(X, mle(weights[0], fs, X), color="red", label=str("Just right, Lambda = " + str(learning_rate[0])))
    ls.append(legend)
    legend, = plt.plot(X, mle(weights[1], fs, X), color="blue", label=str("Under-fitting, Lambda = " + str(learning_rate[1])))
    ls.append(legend)
    legend, = plt.plot(X, mle(weights[2], fs, X), color="green", label=str("Over-fitting, Lambda = " + str(learning_rate[2])))
    ls.append(legend)

    plt.ylim(top=1.8)
    plt.ylim(bottom=-1.5)
    plt.legend(handles=ls)
    plt.show()


if __name__ == '__main__':
    # f1a()
    # f1b()
    # f1c()
    f2b()

    '''
    X, Y = get_db(25, (-0.3, 1.3))
    mle(Y, get_fs_poly(11), X)
    '''
