import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as anim

q = 1
m = 1

def plotting(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def magnetic_field(t, x, y, z):
    B0 = 1.0
    return np.array([B0*np.exp(-t*x), B0 * np.cos(t*y), B0 * np.sin(t*z)])

def equations_of_motion(t, y):
    x, y, z, vx, vy, vz = y
    B = magnetic_field(t, x, y, z) # F = g * [V, B]

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = (q / m) * (vy * B[2] - vz * B[1])
    dvydt = (q / m) * (vz * B[0] - vx * B[2])
    dvzdt = (q / m) * (vx * B[1] - vy * B[0])

    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]

initial_conditions = [0, 0, 0, 1, 0, 0]

t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 500)

solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval)
x, y, z = solution.y[:3]

plotting(x, y, z)

fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')
line, = axis.plot([], [], [], lw=1)
axis.set_xlabel('X')
axis.set_ylabel('Y')
axis.set_zlabel('Z')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def animate(frame_number):
    n = frame_number
    line.set_data(x[:n], y[:n])
    line.set_3d_properties(z[:n])

    axis.scatter(x[frame_number], y[frame_number], z[frame_number], s=0.2, marker='o', c='b')
    return line,

N = len(x)
ani = anim.FuncAnimation(fig, animate, frames=N, init_func=init, interval=20, blit=True)
ani.save('fuck.gif', writer='imagemagick')
plt.show()