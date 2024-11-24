import numpy as np
import matplotlib.pyplot as plt


# Исходная система (например, модель дифференциального уравнения)
def original_system(t, state, params):
    # Пример системы: dx/dt = -ax + b, dy/dt = c * x - d * y
    x, y = state
    a, b, c, d = params
    dxdt = -a * x + b
    dydt = c * x - d * y
    return np.array([dxdt, dydt])


# Линеаризованная система (приближенная)
def linearized_system(t, state, params):
    # Линеаризация исходной системы вокруг фиксированной точки (например, x=0, y=0)
    x, y = state
    a, b, c, d = params
    dxdt = -a * x + b
    dydt = c * x - d * y
    return np.array([dxdt, dydt])


# Численное решение с методом Эйлера
def euler_method(system, t_span, y0, params, dt=0.01):
    t_values = np.arange(t_span[0], t_span[1], dt)
    state_values = np.zeros((len(t_values), len(y0)))
    state_values[0] = y0

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        state = state_values[i - 1]
        state_values[i] = state + dt * system(t, state, params)

    return t_values, state_values


# Заданные параметры системы
params = [1, 1, 1, 1]  # a, b, c, d
y0 = [1, 0]  # начальные значения
t_span = [0, 10]  # временной интервал

# Численное решение исходной и линеаризованной системы
t, state_original = euler_method(original_system, t_span, y0, params)
_, state_linearized = euler_method(linearized_system, t_span, y0, params)

# Построение графиков
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, state_original[:, 0], label='x (Исходная система)')
plt.plot(t, state_linearized[:, 0], label='x (Линеаризованная система)', linestyle='dashed')
plt.title('Переменная x')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, state_original[:, 1], label='y (Исходная система)')
plt.plot(t, state_linearized[:, 1], label='y (Линеаризованная система)', linestyle='dashed')
plt.title('Переменная y')
plt.legend()

plt.show()
