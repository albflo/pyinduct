import pickle
import numpy as np
import sympy as sp
from sympy.utilities.autowrap import ufuncify

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 3

with open("symb_test_N={}.pkl".format(N), "rb") as f:
    data = pickle.load(f)

inputs, state, rhs_vec, rhs_jac = sp.sympify(data)

print(inputs)
print(state)
print(rhs_vec)

rhs_cb = sp.lambdify(args=[inputs, state], expr=rhs_vec, modules="numpy")
jac_cb = sp.lambdify(args=[inputs, state], expr=rhs_jac, modules="numpy")

q_dt = rhs_cb(np.array([0 for entry in inputs]),
              np.array([0 for entry in state]))
print(q_dt)


def inputs(t):
    return np.array([0, 0])


def rhs(t, y):
    u = inputs(t)
    y_dt = rhs_cb(u, y)
    return y_dt


def jac(t, y):
    u = inputs(t)
    jac = jac_cb(u, y)
    return jac


y0 = np.array([-10, -10, 10, 10, .5])
res = solve_ivp(rhs, (0, 100), y0, jac=jac, method="BDF")

f, ax1 = plt.subplots(1, 1, sharex=True)
for idx, point in enumerate(res.y):
    ax1.plot(res.t, point, label="x_{}".format(idx))
ax1.legend()
ax1.grid()

z_grid = np.linspace(0, 1, num=N)
f, ax2 = plt.subplots(1, 1, sharex=True)
for idx, point in enumerate(res.y.T[::10]):
    ax2.plot(z_grid, np.hstack((point[:2], 0)), label="T_{}".format(idx))
ax2.legend()
ax2.grid()

# f = plt.figure()
# for idx, point in enumerate(res.y):
#     plt.plot(res.t, point, label="x_{}".format(idx))
#
plt.show()
