import random
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def setup():
    SEED = 128
    random.seed(SEED)
    np.random.seed(SEED)

def create_linear_regression_data(b_0, b_1, mu=0, sigma=1, start=0, size=1000):
    x = np.arange(start, size + 1)
    errors = np.random.normal(mu, sigma, size=len(x))
    y = b_0 + b_1 * x + errors
    return x, y

def plot_scatter(x, y):
    plt.scatter(x, y, s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatterplot de datos simulados")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def gradient_descent(xs, ys, f, dx=0.01, max_ciclos=1000000, tol=1e-6, x_0=1, y_0=1):
    x_cur = x_0
    x_new = 0
    y_cur = y_0
    y_new = 0

    count = 0

    found = False
    max_reached = False

    grad_fx = sp.lambdify((xs, ys), sp.diff(f, xs), 'numpy')
    grad_fy = sp.lambdify((xs, ys), sp.diff(f, ys), 'numpy')

    while (not found and not max_reached):

        gx = grad_fx(x_cur, y_cur)
        gy = grad_fy(x_cur, y_cur)

        x_new = x_cur - dx * gx
        y_new = y_cur - dx * gy

        count += 1

        if count >= max_ciclos:
            max_reached = True

        v_new = np.array([x_new, y_new])
        v_cur = np.array([x_cur, y_cur])

        distance = np.linalg.norm(v_new - v_cur)

        if distance < tol:
            found = True

        x_cur = x_new
        y_cur = y_new

    return x_new, y_new


def matrix_analytic(x, y):
    x = np.array(x)
    y = np.array(y)

    n = len(y)

    ones = np.ones((n, 1))
    x_col = x.reshape(n, 1)

    X = np.column_stack((ones, x_col))

    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    return beta[0], beta[1]


def main():
    setup()

    beta_0 = 1
    beta_1 = 2

    x, y = create_linear_regression_data(beta_0, beta_1)
    plot_scatter(x, y)


    xs, ys = sp.symbols('xs ys')

    residuals = [(y[i] - (xs + ys*x[i]))**2 for i in range(len(x))]
    f = sp.Add(*residuals)

    numeric_beta_0, numeric_beta_1 = gradient_descent(xs, ys, f, dx=1e-12)

    analytic_beta_0, analytic_beta_1 = matrix_analytic(x, y)

    print(f"Actual: b0={beta_0}, b1={beta_1}")
    print(f"Numeric (GD): b0={numeric_beta_0}, b1={numeric_beta_1}")
    print(f"Analytic:     b0={analytic_beta_0}, b1={analytic_beta_1}")


if __name__ == "__main__":
    main()
