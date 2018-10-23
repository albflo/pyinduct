import pickle
import sympy as sp
# import symengine as sp
import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
from matplotlib import pyplot as plt


class Controller(pi.SimulationInput):

    def __init__(self, gain, idx):
        super().__init__("BC")
        self.gain = gain
        self.idx = idx

    def _calc_output(self, **kwargs):
        """ Top notch boundary feedback """
        ret = -self.gain * kwargs["weights"][self.idx]
        return dict(output=ret)


# approximation order
N = 5

temp_dom = pi.Domain((0, 5), num=100)

# spatial domain
spat_bounds = (0, 1)
spat_dom = pi.Domain(spat_bounds, num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]

# define system inputs
u1 = ss.get_input()
u2 = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x = ss.get_field_variable(z, t)

# define symbols for test functions
phi_k = ss.get_test_function(z)

# define parameters
alpha = sp.symbols("alpha", real=True)
param_list = [
    (alpha, .1),
    # parameters for approximation of nonlinearities
    # ("enable_approx", True),
    ("enable_approx", False),
    # ("approx_mode", "series"),
    ("approx_mode", "pointwise"),
    ("approx_pos", .5),
    ("approx_order", 2),
]
ss.register_parameters(*param_list)

# define boundaries
boundaries = [
    # sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[0], evaluate=False), -10),
    # sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[1], evaluate=False), 10),
    sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[0], evaluate=False), -u1),
    sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[1], evaluate=False), u2),
]

# define approximation basis
if 1:
    nodes = pi.Domain(spat_bounds, num=N)
    fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
else:
    fem_base = ss.create_lag1st_base(z, spat_bounds, N)

pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x_approx = ss.create_approximation(z, "fem", boundaries)

# define the initial conditions for each approximation
ics = {
    x_approx: 10
    # x_approx: 10 *z
    # x_approx: 1e-1 * sp.sin(z*sp.pi)
}

# define the system inputs and their mapping
input_dict = {
    u1: Controller(1, 0),
    u2: Controller(1, -1)
}

if 0:
    state0 = x_approx.approximate_function(x0)
    a = x_approx.get_spatial_approx(state0)
    vals = np.linspace(*spat_bounds)
    plt.plot(vals, a(vals))
    plt.show()

# some spatial dependent coefficients
# a0 = sp.sin(z)

# some variants for nonlinearities, most of them unstable
# a0 = 0
# a0 = 1
# a0 = (1 + 10 * x)
# a0 = x
# a0 = x**2
a0 = sp.cos(x)
# a0 = sp.cos(t)
# a0 = -2*x**2 + sp.exp(x)

# define the variational formulation for both phases
weak_form = [
    ss.InnerProduct(x.diff(t), phi_k, spat_bounds)
    - alpha * ((x.diff(z)*phi_k).subs(z, 1)
               - (x.diff(z)*phi_k).subs(z, 0)
               - ss.InnerProduct(x.diff(z), phi_k.diff(z), spat_bounds))
    - ss.InnerProduct(a0 * x, phi_k, spat_bounds)
]
sp.pprint(weak_form, num_columns=200)

# define which symbol shall be approximated by which approximation
x_test = x_approx.base
rep_dict = {
    x: x_approx,
    phi_k: x_test,
}

results = ss.simulate_system(weak_form, rep_dict, input_dict, ics, temp_dom,
                             spat_dom)

with open("diff_sys_approx.pkl", "wb") as f:
    pickle.dump(results, f)


win = pi.PgAnimatedPlot(results)
wins = pi.PgSurfacePlot(results[0])
pi.show()
