from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def model(v, t):
    ror = 1
    V = 1
    g = 10
    row = 1
    Cd = 1
    A = 1
    m = 1
    ma = 1

    dvdt = (ror * V * g - row * V * g - 1 / 2 * Cd * row * A * v * v) / (m + ma)
    return dvdt


#
# S = 1
v0 = [0, 0.3, 1]

t = np.linspace(0, 20, 200)
result = odeint(model, v0,  t)  # v0 is Initial condition

fig, ax = plt.subplots()
ax.plot(t, result[:, 0], label='v0=0')
ax.plot(t, result[:, 1], label='v0=0.3')
ax.plot(t, result[:, 2], label='v0=1')
ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('Rp')

plt.show()
