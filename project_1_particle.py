import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as anim


def magnetic_field(t: float, x: float, y: float, z: float) -> list:
    """function, which returns parameters of magnetic field in given time and coordinates"""
    B0: float = 1.0
    return np.array([B0 * np.exp(-x), B0 * np.cos(y), B0 * np.sin(z)])


def equations_of_motion(t: float, coordinates: list) -> list:
    """system of differential equations, which describe particle's motion"""
    x, y, z, vx, vy, vz = coordinates
    B = magnetic_field(t, x, y, z)  # F = g * [V, B]

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = (q / m) * (vy * B[2] - vz * B[1])
    dvydt = (q / m) * (vz * B[0] - vx * B[2])
    dvzdt = (q / m) * (vx * B[1] - vy * B[0])

    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]


def plotting(x: float, y: float, z: float) -> None:
    """plot based on the given array of points"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def animate(n: int):
    ax.clear()

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    ax.set_title('Particle Deposition in the Magnetic Field\nTime: 10 sec')

    ax.scatter(x[n], y[n], z[n], s=2, color='b')

    return line


if __name__ == "__main__":

    q: float = 1.0  # charge
    m: float = 1.0  # mass

    initial_conditions = [0, 0, 0, 1, 0, 0]

    t_span: list = [0, 10]  # time limits
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval)
    x, y, z = solution.y[:3]

    plotting(x, y, z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line = ax.plot([], [], [], lw=3)

    N: int = len(x)
    ani = anim.FuncAnimation(fig, animate, frames=N, interval=20, blit=True)
    ani.save('1_particle.gif', writer='imagemagick')
