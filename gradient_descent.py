import matplotlib.pyplot as plt
import numpy as np

# Definition of the particular Ackley function we are working with
# The Ackley function can be separated into two major terms, seen here
# And then two smaller terms which can just be summed
def f(x, y):
    """
    f(x, y) = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2pi*x) + cos(2pi*y))) + e + 20
    """
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + np.e + 20


def grad_f(x, y):
    """Computes the analytical gradient vector [df/dx, df/dy] at (x, y)."""

    # The denominator of the derivative of the first term
    # This part: vvvvvvvvvvvvvvvvvvvvvvv
    # u = -0.2 * sqrt(0.5 * (x^2 + y^2))
    # df1/dx = -20 * exp(u) * du/dx
    denom = np.sqrt(0.5 * (x ** 2 + y ** 2))

    # At the origin, this will cause a division by zero lol
    # We can expressly say that, when this happens, the derivatives will be zero
    # (Can also be a small number because of floating-point errors)
    if denom < 1e-10:
        deriv_term1_x = 0
        deriv_term1_y = 0
    else:
        exp_u = np.exp(-0.2 * denom)
        # 2 * x * exp(u) / denom
        deriv_term1_x = 2 * x * exp_u / denom
        deriv_term1_y = 2 * y * exp_u / denom

    # Now we do pretty much the same thing but for the second term
    # pi * sin(2pi*x) * exp(v)
    exp_v = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    deriv_term2_x = np.pi * np.sin(2 * np.pi * x) * exp_v
    deriv_term2_y = np.pi * np.sin(2 * np.pi * y) * exp_v

    # Sum the partial derivatives of the corresponding terms to obtain the full partial derivatives
    gx = deriv_term1_x + deriv_term2_x
    gy = deriv_term1_y + deriv_term2_y
    return gx, gy

def main():
    # Parameters for the Gradient Descent Loop
    x_0 = 0.8
    y_0 = 0.8
    learning_rate = 0.01
    iterations = 100
    path = [(x_0, y_0)]

    for i in range(iterations):
        gx, gy = grad_f(x_0, y_0)
        x_0 -= learning_rate * gx
        y_0 -= learning_rate * gy
        path.append((x_0, y_0))

    # Obtain the final converged point
    # We also stored the path data to make it look pretty
    min_x, min_y = path[-1]
    min_z = f(min_x, min_y)
    hx, hy = zip(*path)
    hz = [f(px, py) for px, py in path]

    ### Plotting Stuff

    # Define the grid
    x_grid = np.linspace(-1.5, 1.5, 100)
    y_grid = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    # 3D plot in its own figure
    fig3d = plt.figure(figsize=(12, 8))
    axes_one = fig3d.add_subplot(1, 1, 1, projection='3d')
    axes_one.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=3, cstride=3)

    # Plot the gradient descent path
    axes_one.plot(hx, hy, hz, 'r-', linewidth=2, label='GD Path')
    axes_one.scatter(hx[0], hy[0], hz[0], color='yellow', s=50, label='Start (1,1)')

    # Add a dot at the local minimum to see where we ended up
    z_min_overall = np.min(Z)

    # And the line from the starting point to the final point
    line_x = [min_x, min_x]
    line_y = [min_y, min_y]
    line_z = [z_min_overall, min_z]
    axes_one.plot(line_x, line_y, line_z, 'red', linewidth=3, label='Local Minimum')
    axes_one.scatter(min_x, min_y, min_z, color='red', marker='o', s=50)  # Marker on the surface

    # Global minimum @ (0, 0)
    # Add a little star at the local minimum to see how far off we are
    global_min_z = f(0, 0)
    axes_one.scatter(0, 0, global_min_z, color='green', marker='*', s=200, label='Global Min (0,0)')

    axes_one.set_title(r'Ackley Function $f(x, y)$ for Vectors in Two Dimensions', fontsize=12)
    axes_one.set_xlabel('x')
    axes_one.set_ylabel('y')
    axes_one.set_zlabel('f(x, y)')
    axes_one.view_init(elev=30, azim=-120)
    axes_one.legend()

    # Save the 3D plot
    fig3d.tight_layout()
    fig3d.savefig("/home/ayimany/very-cool-plot-3d.png")

    # 2D contour plot in its own figure
    fig2d = plt.figure(figsize=(12, 8))
    axes_two = fig2d.add_subplot(1, 1, 1)
    contour = axes_two.contourf(X, Y, Z, levels=30, cmap='viridis')
    fig2d.colorbar(contour, ax=axes_two, label='f(x, y)')

    # Plot the gradient descent path in 2D
    axes_two.plot(hx, hy, 'r.-', label='Gradient Descent Path')
    axes_two.scatter(hx[0], hy[0], color='yellow', s=50, label='Start (1,1)')
    axes_two.scatter(min_x, min_y, color='red', marker='o', s=50, label='Local Min')
    axes_two.scatter([0], [0], color='green', marker='*', s=200, label='Global Min (0,0)')

    axes_two.set_title('Contour Plot of the Ackley Function', fontsize=14)
    axes_two.set_xlabel('x')
    axes_two.set_ylabel('y')
    axes_two.legend()

    # Make it look functional
    axes_two.set_anchor('C')
    axes_two.margins(0)

    fig2d.tight_layout()
    axes_two.set_aspect('equal', adjustable='box')

    # Save the 2D figure separately
    fig2d.savefig(f"/home/ayimany/very-cool-plot-2d.png")

    print(f"The minimum reached is: ({min_x}, {min_y}, {min_z})")

if __name__ == '__main__':
    main()
