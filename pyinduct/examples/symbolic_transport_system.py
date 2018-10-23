import sympy as sp
# import symengine as sp
import numpy as np
import pyinduct as pi
import pyinduct.sym_simulation as ss
from matplotlib import pyplot as plt

# approximation order
N = 10

temp_dom = pi.Domain((0, 5), num=100)

# spatial domain
spat_bounds = (0, 1)
spat_dom = pi.Domain(spat_bounds, num=100)

# define temporal and spatial variables
t = ss.time
z = ss.space[0]

# define system inputs
u1 = ss.get_input()

# define symbols of spatially distributed and lumped system variables
x = ss.get_field_variable(z, t)

# define symbols for test functions
phi_k = ss.get_test_function(z)

# define parameters
param_list = [
    # ("enable_approx", True),
    ("approx_pos", .5),
    ("approx_order", 2),
]
ss.register_parameters(*param_list)

# define boundaries
boundaries = [
    sp.Eq(sp.Subs(x, z, spat_bounds[0], evaluate=False), u1),
    # sp.Eq(sp.Subs(x.diff(z), z, spat_bounds[0], evaluate=False), u1),
]

nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem", fem_base)

# create approximations, homogenizing where needed
x_approx = ss.create_approximation(z, "fem", boundaries)

# define the initial conditions for each approximation
ics = {
    # x_approx: 1e-1 * sp.sin(z*sp.pi)
    # x_approx: 10 * sp.cos(.5*z*sp.pi),
    x_approx: 10,
    u1: 10
}

# define the system inputs and their mapping
input_map = {
    # u1: pi.ConstantTrajectory(const=0)
    u1: pi.InterpolationTrajectory(temp_dom, 10*np.cos(2*temp_dom.points))
}

if 0:
    # state0 = x_approx.approximate_function(ics[x_approx], [ics[u1]])
    # a = x_approx.get_spatial_approx(state0, [ics[u1]])
    state0 = x_approx.approximate_function(ics[x_approx])
    a = x_approx.get_spatial_approx(state0)
    vals = np.linspace(*spat_bounds, num=1000)
    plt.plot(vals, a(vals))
    plt.show()
    quit()

v = -1

# define the variational formulation for both phases
weak_form = [
    ss.InnerProduct(x.diff(t), phi_k, spat_bounds) - v * (
        # ss.InnerProduct(x.diff(z), phi_k, spat_bounds)
        (x * phi_k).subs(z, 1)
        - (x * phi_k).subs(z, 0)
        - ss.InnerProduct(x, phi_k.diff(z), spat_bounds)
    )
]
sp.pprint(weak_form, num_columns=200)

# define which symbol shall be approximated by which approximation
rep_dict = {
    x: x_approx,
    phi_k: x_approx.base,
}


def controller(t, weights):
    """ Top notch boundary feedback """
    # k = 1e2
    k = 0
    return -k * weights[0]


results = ss.simulate_system(weak_form, rep_dict, input_map, ics,
                             temp_dom, spat_dom)


# plots = []
# for res in results:
#     p = plt.plot(res.input_data[1].points, res.output_data[0, :])
#     plots.append(p)
win = pi.PgAnimatedPlot(results)
wins = pi.PgSurfacePlot(results[0])
pi.show()
