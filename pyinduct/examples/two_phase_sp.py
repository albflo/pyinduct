import pickle
import numpy as np
import sympy as sp

import pyinduct as pi
import pyinduct.sym_simulation as ss

import pyqtgraph as pg

# approximation order
N = 20

# spatial domains
spat_dom = pi.Domain((0, 1), num=N)
zb_dom = (0, 1)

# temporal domain
temp_dom = pi.Domain((0, 5), num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]
# z1, z2 = ss.space[:2]

# define system inputs
u1 = ss.get_input()
u2 = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x1 = ss.get_field_variable(z, t)
x2 = ss.get_field_variable(z, t)
gamma = sp.symbols("gamma", cls=sp.Function)(t)

# define symbols for test functions
phi1_k = ss.get_test_function(z)
phi2_k = ss.get_test_function(z)

# define parameters
alpha, Gamma, k, rho, L, Tm = sp.symbols(("alpha:2",
                                          "Gamma:2",
                                          "k:2",
                                          "rho:3",
                                          "L",
                                          "T_m"), real=True)
param_list = [
    (alpha[0], .591/(2.05*.91)),
    (alpha[1], 2.2/(4.19*1)),
    (Gamma[0], zb_dom[0]),
    (Gamma[1], zb_dom[1]),
    (k[0], .591),
    (k[1], 2.2),
    (rho[0], .91),
    (rho[1], 1),
    (rho[2], 1),  # rho(Tm)
    (L, 334e3),
    (Tm, 0),
]
ss.register_parameters(*param_list)

# define boundaries
boundaries_x1 = [
    sp.Eq(sp.Subs(x1.diff(z), z, 0), -(gamma - Gamma[0]) / k[0] * u1),
    sp.Eq(sp.Subs(x1, z, 1), Tm),
]
boundaries_x2 = [
    sp.Eq(sp.Subs(x2.diff(z), z, 0), (gamma - Gamma[1]) / k[1] * u2),
    sp.Eq(sp.Subs(x2, z, 1), Tm),
]

# define approximation basis
if 1:
    nodes = pi.Domain(zb_dom, num=N)
    fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
else:
    fem_base = ss.create_lag1ast_base(z, zb_dom, N)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x1_approx = ss.create_approximation(z, "fem", boundaries_x1)
x2_approx = ss.create_approximation(z, "fem", boundaries_x2)
gamma_approx = ss.get_weight()

# define the initial conditions for each approximation
ics = {
    x1_approx: -10 * (1-z),
    x2_approx: 10 * (1-z),
    gamma_approx: .5
}

# define the system inputs and their mapping
input_map = {
    u1: pi.ConstantTrajectory(-500),
    u2: pi.ConstantTrajectory(0),
}

# define the variational formulation for both phases
equations = []
for idx, (x, u, phi) in enumerate(zip([x1, x2],
                                      [u1, u2],
                                      [phi1_k, phi2_k])):
    beta = (gamma - Gamma[idx])
    expr = (
        ss.InnerProduct(x.diff(t), phi, zb_dom)
        - alpha[idx] / beta ** 2 * (
            (x.diff(z) * phi).subs(z, 1)
            - (x.diff(z) * phi).subs(z, 0)
            - ss.InnerProduct(x.diff(z), phi.diff(z), zb_dom)
        )
        - gamma.diff(t) / beta * ss.InnerProduct(z * x.diff(z), phi, zb_dom)
    )
    equations.append(expr)

# define the ode for the phase boundary
g_exp = (k[0] / (gamma - Gamma[0]) * x1.diff(z).subs(z, 1)
         - k[1] / (gamma - Gamma[1]) * x2.diff(z).subs(z, 1)
         - gamma.diff(t) * rho[2] * L
         )
equations.append(g_exp)
sp.pprint(equations)

# create test functions, easy due to Galerkin principle
x1_test = x1_approx.base
x2_test = x2_approx.base

approx_map = {
    x1: x1_approx,
    x2: x2_approx,
    phi1_k: x1_test,
    phi2_k: x2_test,
    gamma: gamma_approx
}

results = ss.simulate_system(equations, approx_map, input_map, ics, temp_dom, spat_dom)

if 0:
    data = str(inputs), str(state), str(sys)
    with open("symb_test_N={}.pkl".format(N), "wb") as f:
        pickle.dump(data, f)
    quit()

# post processing

if 0:
    # build state transformation
    zeta_trafo = [(z - Gamma[idx])/(gamma_approx - Gamma[idx]) for idx in range(2)]
    # approximations = [appr.subs(z, trafo) for appr, trafo in
    #                   zip([x1_approx, x2_approx], zeta_trafo)]
else:
    approximations = [x1_approx, x2_approx]

# gamma_sim = res_weights.y[-1]
plots = []
for res in results:
    p = pg.plot(res.input_data[1].points, res.output_data[0, :])
    plots.append(p)

win = pi.PgAnimatedPlot(results)
win1 = pi.PgSurfacePlot(results[0])
win2 = pi.PgSurfacePlot(results[1])
pi.show()


