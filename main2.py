import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# Constant values
g = 9.81
l = 1.0

# 1. Реализовать возможность прямого численного моделирования для исходной и линеаризованной системы
#    при произвольных параметрах и различных начальных значениях переменных, определяющих состояние системы

def rhs(y, t):
    phi_, dphi_, theta_, dtheta_, = y
    dphi = dphi_
    ddphi = -2 * dphi_ * dtheta_ / (np.tan(theta_) + 1e-12)
    dtheta = dtheta_
    ddtheta = np.sin(theta_) * np.cos(theta_) * dphi_**2 - (g / l) * np.sin(theta_)
    return [dphi, ddphi, dtheta, ddtheta]

def linear_rhs(y, t):
    phi, dphi, theta, dtheta = y
    return [dphi, 0, dtheta, -g/l * theta]

# 2. Реализовать функцию, которая для фиксированного набора параметров линеаризованной системы
#    определяет начальные значения, при которых будут реализовываться нормальные колебания.

def find_normal_mode_initial_conditions(A):

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        print("Матрица не диагонализируема. Нормальные колебания не могут быть найдены.")
        return None

    initial_conditions = []
    for eigenvector in eigenvectors.T:
        initial_conditions.append(np.real(eigenvector))

    return initial_conditions

A = np.array([[0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, -g/l, 0],])
y0s = find_normal_mode_initial_conditions(A)
y0s = y0s[:-2] # dropping the last two because they're zero-ish
print(y0s)

# 3. На основе прямого численного моделирования построить временные реализации переменных,
#    определяющих состояние линеаризованной системы в режиме нормальных колебаний.
#    Сравнить их с динамикой исходной системы при аналогичных начальных условиях.

# Plotting comparative time series
def plot_time_series(y0s, l, t_max=5, num_points=1000):

  fig, axes = plt.subplots(2, 2, figsize=(15, 10))

  for k, y0 in enumerate(y0s):
    t = np.linspace(0, t_max, num_points)
    print(y0)
    sol = odeint(rhs, y0, t)
    sol_lin = odeint(linear_rhs, y0, t)
    sol = sol.T
    sol_lin = sol_lin.T

    xlabel = 't'
    ylabels = [r'$\phi$', r'$\dot{\phi}$', r'$\theta$', r'$\dot{\theta}$']

    # sol and sol_lin order of variables: [phi, dphi, theta, dtheta]

    n = 2 # matrix dimension

    for i in range(n):
      for j in range(n):
        axes[i, j].plot(t, sol[i + n*j], label=f"Initial {k+1}", color='red')
        axes[i, j].plot(t, sol_lin[i + n*j], linestyle='--', color='orange', label=f"Linear initial {k+1}")

        axes[i, j].set_xlabel('t')
        axes[i, j].set_ylabel(ylabels[i + n*j])
        axes[i, j].legend()

# y0s = [[0, 10, np.pi / 4, 0]]
plot_time_series(y0s, l)

# Plotting 3D dynamics of the pendulum
# Дорисовтаь точку в начале
def plot_3d_dynamics(y0s, l, t_max=5, num_points=1000, rhs=rhs):
  fig = plt.figure(figsize=(8, 8))
  axes = fig.add_subplot(111, projection='3d')

  for k, y0 in enumerate(y0s):
    t = np.linspace(0, t_max, num_points)
    solution = odeint(rhs, y0, t)
    phi, _, theta, _ = solution.T

    x = l * np.sin(theta) * np.cos(phi)
    y = l * np.sin(theta) * np.sin(phi)
    z = -l * np.cos(theta)
    # print(x, y, z)

    # Start point
    start_x = l * np.sin(y0[2]) * np.cos(y0[0])
    start_y = l * np.sin(y0[2]) * np.sin(y0[0])
    start_z = -l * np.cos(y0[2])

    axes.plot(x, y, z, label=f"Initial {k+1}")
    axes.scatter(start_x, start_y, start_z, s=50)

  axes.set_xlabel('x')
  axes.set_ylabel('y')
  axes.set_zlabel('z')
  axes.set_title('Pendulum 3D trajectories')
  axes.legend()

  plt.show()

# Initials in [phi, dphi, theta, dtheta] order
# y0s = [[0, 10, np.pi / 4, 0], [0, 5, np.pi/2, 0], [0, 0, np.pi / 4, 0]]

plot_3d_dynamics(y0s, l, 3)