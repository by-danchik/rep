import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as anim
from random import uniform


def magnetic_field(t: float, x: float, y: float, z: float) -> list:
    #  function, which returns parameters of magnetic field in given time and coordinates
    B0 = 1.0
    return np.array([B0 * np.exp(-t), B0 * np.cos(t), B0 * np.sin(t)])


def equations_of_motion(t: float, coordinates: list) -> list:
    # system of differential equations, which describe particle's motion
    x, y, z, vx, vy, vz = coordinates
    B = magnetic_field(t, x, y, z)  # F = g * [V, B]

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = (q / m) * (vy * B[2] - vz * B[1])
    dvydt = (q / m) * (vz * B[0] - vx * B[2])
    dvzdt = (q / m) * (vx * B[1] - vy * B[0])

    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]


def animate(n: int):
    ax.clear()

    ax.set_xlim(xlim_left, xlim_right)
    ax.set_ylim(ylim_left, ylim_right)
    ax.set_zlim(zlim_left, zlim_right)
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    ax.set_title('20 Particles Deposition in the Magnetic Field\nTime: 10 sec')

    for i in range(num_particles):
        ax.scatter(x[i][n], y[i][n], z[i][n], s=2, color='b')

    return line


q = 1
m = 1
num_particles = 20


initial_coordinates = [[0] * 3 for i in range(num_particles)]
for i in range(num_particles):
    initial_coordinates[i][0] = uniform(-0.2, 0.2)
    initial_coordinates[i][1] = uniform(-0.2, 0.2)
    initial_coordinates[i][2] = uniform(-0.2, 0.2)


initial_conditions = []
for i in range(num_particles):
    initial_conditions.append([initial_coordinates[i][0],
                               initial_coordinates[i][1],
                               initial_coordinates[i][2], 1, 0, 0])


t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 500)


x = [[0] for i in range(num_particles)]
y = [[0] for i in range(num_particles)]
z = [[0] for i in range(num_particles)]


for i in range(num_particles):
    solution = solve_ivp(equations_of_motion, t_span, initial_conditions[i], t_eval=t_eval)
    x[i], y[i], z[i] = solution.y[:3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line = ax.plot([], [], [], lw=3)


xlim_left = min(map(min, x))
xlim_right = max(map(max, x))
ylim_left = min(map(min, y))
ylim_right = max(map(max, y))
zlim_left = min(map(min, z))
zlim_right = max(map(max, z))


N = len(x[0])
ani = anim.FuncAnimation(fig, animate, frames=N, interval=20, blit=True)
ani.save('N_particle.gif', writer='imagemagick')
