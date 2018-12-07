import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as sy
import utils
import visu


# spatial approximation order
N = 20

# system parameters
load_mass = 1e0
a_rho = 1e2
gravity = 9.81
chain_length = 1
force = 1e1

# temporal domain
T = 10
temp_dom = pi.Domain((0, T), step=0.01)

# spatial domain
spat_bounds = (0, chain_length)
spat_dom = pi.Domain(spat_bounds, step=0.5)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.SimulationInputSum([
    utils.ConstantInput(force)
]))

# variables
var_pool = sy.VariablePool("heavy chain")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")

# input variable which holds a pyinduct.SimulationInputWrapper
# as implemented function needs a unique  variable  from which
# they depend, since they are called with a bunch of
# arguments during simulation
input_arg = var_pool.new_symbol("input_arg", "simulation input argument")
u = var_pool.new_implemented_function("u", (input_arg,), input_, "input")
input_vector = sp.Matrix([u])

# define approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("complete_base", pi.Base(list(fem_base.fractions)*2))
pi.register_base("fem_base", fem_base)
init_funcs_w = var_pool.new_implemented_functions(
    ["phi_{w" + str(i) + "}" for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions w")
init_funcs_v = var_pool.new_implemented_functions(
    ["phi_{v" + str(i) + "}" for i in range(N)], [(z,)] * N,
    fem_base.fractions, "initial functions v")
test_funcs_half_w = sp.Matrix(var_pool.new_implemented_functions(
    ["psi_{w" + str(i) + "}" for i in range(N)], [(z,)] * N,
    fem_base.fractions, "test functions w"))
test_funcs_half_v = sp.Matrix(var_pool.new_implemented_functions(
    ["psi_{v" + str(i) + "}" for i in range(N)], [(z,)] * N,
    fem_base.fractions, "test functions v"))

# build approximation
weights_w = sp.Matrix(var_pool.new_functions(
    ["c_{w" + str(i) + "}" for i in range(N)], [(t,)] * N, "approximation weights for w"))
weights_v = sp.Matrix(var_pool.new_functions(
    ["c_{v" + str(i) + "}" for i in range(N)], [(t,)] * N, "approximation weights for v"))
w_approx = sum([c * phi for c, phi in zip(weights_w, init_funcs_w)])
v_approx = sum([c * phi for c, phi in zip(weights_v, init_funcs_v)])
sy.pprint(w_approx, "approximation of w", N)
sy.pprint(v_approx, "approximation of v", N)

# complete weights vector and set of test function
weights = sp.Matrix.vstack(weights_w, weights_v)
test_funcs_w = sp.Matrix.vstack(test_funcs_half_w, test_funcs_half_v * 0)
test_funcs_v = sp.Matrix.vstack(test_funcs_half_w * 0, test_funcs_half_v)
sy.pprint(test_funcs_w, "test functions of w", N)
sy.pprint(test_funcs_v, "test functions of v", N)

# subs param variables
alpha_0 = 1
alpha_1 = alpha_0*load_mass/a_rho
# project on test functions
projections = list()
limits = (z, spat_bounds[0], spat_bounds[1])
tau = gravity * (load_mass + a_rho*z)
for psi_w, psi_v in zip(test_funcs_w, test_funcs_v):
    projections.append(
        sp.Integral(sp.diff(w_approx, t) * psi_w, limits) * 1
        + sp.Integral(sp.diff(v_approx, t) * psi_v, limits) * alpha_0
        - sp.Integral(v_approx * psi_w, limits) * 1
        + sp.diff(v_approx, t).subs(z, 0) * psi_v.subs(z, 0) * alpha_1
        + sp.Integral(sp.diff(w_approx, z) * tau * sp.diff(psi_v, z), limits) * alpha_0/a_rho
        - u * psi_v.subs(z, chain_length) * alpha_0/a_rho
    )
projections = sp.Matrix(projections)
sy.pprint(projections, "projections", N)

# evaluate integrals
projections = sy.evaluate_integrals(projections)
sy.pprint(projections, "evaluated integrals in projections", N)

# evaluate remaining implemented functions
projections = sy.evaluate_implemented_functions(projections)
sy.pprint(projections, "evaluated projections", N)

# initial conditions
init_samples = np.zeros(len(weights))

# derive rhs and simulate
rhs = sy.derive_first_order_representation(projections, weights, input_vector,
                                           # mode="sympy.solve")
                                           mode="sympy.linear_eq_to_matrix")
sy.pprint(rhs, "right hand side of the discretization", N)

# use numpy.dot to speed up the simulation (compare / run without this line)
rhs = sy.implement_as_linear_ode(rhs, weights, input_vector)

# simulate
_, q = sy.simulate_system(
    rhs, weights, init_samples, "complete_base", input_vector, t, temp_dom)

# visualization
data = pi.get_sim_result("fem_base", q, temp_dom, spat_dom, 0, 0)
# win = pi.PgAnimatedPlot(data)

# visu.hc_visualization(data, nodes)

# pi.show()
