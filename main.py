import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.81
l = 1.0

# Исходная система
def nonlinear_system(t, y):
    theta, theta_dot, phi, phi_dot = y
    theta_ddot = np.sin(theta) * np.cos(theta) * phi_dot ** 2 - (g / l) * np.sin(theta)
    phi_ddot = -2 * np.cos(theta) * theta_dot * phi_dot / np.sin(theta)
    return [theta_dot, theta_ddot, phi_dot, phi_ddot]


# Линеаризованная система
def linearized_system(t, y):
    theta, theta_dot, phi, phi_dot = y
    theta_ddot = -(g / l) * phi
    phi_ddot = 0
    return [theta_dot, theta_ddot, phi_dot, phi_ddot]

# Функция для нормальных колебаний
def find_normal_conditions(l):
    # Проверка условия колебаний
    if g / l > 0:
        theta0 = np.pi / 6  # малый угол
        theta_dot0 = 0.0
        phi0 = 0.0
        phi_dot0 = 1.0
        return [theta0, theta_dot0, phi0, phi_dot0]
    else:
        raise ValueError("Параметры не подходят для нормальных колебаний")

def simulate_systems(params, y0, t_span=(0, 10), n_points=1000):
    # Временная сетка
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Численное решение для исходной системы
    sol_nonlinear = solve_ivp(
        nonlinear_system, t_span, y0, t_eval=t_eval, method='RK45'
    )

    # Численное решение для линеаризованной системы
    sol_linear = solve_ivp(
        linearized_system, t_span, y0, t_eval=t_eval, method='RK45'
    )

    return sol_nonlinear, sol_linear


if __name__ == "__main__":
    # 1-2 часть далее

    # y0 = find_normal_conditions(l)  # начальные условия
    y0 = [np.pi / 8, 0.0, 0.0, 1.0]
    t_span = (0, 1000)  # временной интервал

    # Прямое численное моделирование
    sol_nonlinear, sol_linear = simulate_systems(l, y0, t_span)

    # Построение графиков
    plt.figure(figsize=(12, 6))
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], label='Исходная система ($\\theta$)', color='blue')
    plt.plot(sol_linear.t, sol_linear.y[0], label='Линеаризованная система ($\\theta$)', color='orange', linestyle='--')
    plt.title("Прямое численное моделирование")
    plt.xlabel("Время (с)")
    plt.ylabel("Углы (рад)")
    plt.legend()
    plt.grid()
    plt.show()


    # Третья часть далее

    # Временной интервал моделирования
    t_span = (0, 10)  # от 0 до 10 секунд
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # точки для вывода данных

    # Решение системы
    sol_nonlinear = solve_ivp(nonlinear_system, t_span, y0, t_eval=t_eval, method='RK45')
    sol_linear = solve_ivp(linearized_system, t_span, y0, t_eval=t_eval, method='RK45')

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График для исходной системы
    plt.subplot(2, 1, 1)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], label=r'$\theta$ (исходная система)', color='blue')
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label=r'$\phi$ (исходная система)', color='green')
    plt.title("Исходная система (Нормальные колебания)")
    plt.xlabel("Время (с)")
    plt.ylabel("Углы (рад)")
    plt.legend()
    plt.grid()

    # График для линеаризованной системы
    plt.subplot(2, 1, 2)
    plt.plot(sol_linear.t, sol_linear.y[0], label=r'$\theta$ (линеаризованная система)', color='blue')
    plt.plot(sol_linear.t, sol_linear.y[2], label=r'$\phi$ (линеаризованная система)', color='green')
    plt.title("Линеаризованная система (Нормальные колебания)")
    plt.xlabel("Время (с)")
    plt.ylabel("Углы (рад)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Четвертая часть далее

    # Временной интервал моделирования
    t_span = (0, 5)  # от 0 до 50 секунд
    t_eval = np.linspace(t_span[0], t_span[1], 1000)  # точки для вывода данных

    # Решение системы
    sol_nonlinear = solve_ivp(nonlinear_system, t_span, y0, t_eval=t_eval, method='RK45')
    sol_linear = solve_ivp(linearized_system, t_span, y0, t_eval=t_eval, method='RK45')

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # График для исходной системы (биения)
    plt.subplot(2, 1, 1)
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[0], label=r'$\theta$ (исходная система)', color='blue')
    plt.plot(sol_nonlinear.t, sol_nonlinear.y[2], label=r'$\phi$ (исходная система)', color='green')
    plt.title("Исходная система (Биения)")
    plt.xlabel("Время (с)")
    plt.ylabel("Углы (рад)")
    plt.legend()
    plt.grid()

    # График для линеаризованной системы (биения)
    plt.subplot(2, 1, 2)
    plt.plot(sol_linear.t, sol_linear.y[0], label=r'$\theta$ (линеаризованная система)', color='blue')
    plt.plot(sol_linear.t, sol_linear.y[2], label=r'$\phi$ (линеаризованная система)', color='green')
    plt.title("Линеаризованная система (Биения)")
    plt.xlabel("Время (с)")
    plt.ylabel("Углы (рад)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()