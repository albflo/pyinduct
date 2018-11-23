import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as sy
import pyqtgraph as pg


def view_transform(data, node_dist, node_cnt, x_offset, y_offset):
    """
    transforms simulated data into from that is better to view
    :param data: eval_data from simulation run
    :return: transformed data
    """
    print(">>> preparing visualization data")
    t_values = data.input_data[0]
    t_step = t_values[1]
    z_values = data.input_data[1]
    z_step = z_values[1]
    # invert spatially since iteration is easier from 0 to l
    w_values = data.output_data[..., ::-1]

    # gradient method
    w = w_values
    grad = np.gradient(w, t_step, z_step)
    w_dt = grad[0]
    w_dz = grad[1]

    # create eval_data
    w_interp = pi.EvalData(data.input_data, w)
    w_dt_interp = pi.EvalData(data.input_data, w_dt)
    w_dz_interp = pi.EvalData(data.input_data, w_dz, name="spline dz")

    # calculate coordinates
    x_values = np.zeros(w_values.shape)
    y_values = np.zeros(w_values.shape)

    w_len = w_values.shape[1]
    for t_idx, t_val in enumerate(t_values):
        for z_idx, z_val in enumerate(z_values):
            if z_idx == 0:
                x_values[t_idx, z_idx] = x_offset + w_values[t_idx, z_idx]
                y_values[t_idx, z_idx] = y_offset
            else:
                x_values[t_idx, z_idx] = x_values[t_idx, z_idx-1] + \
                                         np.sin(w_dz[t_idx, z_idx-1])*z_step
                y_values[t_idx, z_idx] = y_values[t_idx, z_idx-1] -\
                                         np.cos(w_dz[t_idx, z_idx-1])*z_step

    print("done!")
    return x_values, y_values, [w_interp, w_dt_interp, w_dz_interp]


def hc_visualization(eval_data, nodes, show=True):
    """
    wrapper that draws visualization of the heavy chain, using the simulation output
    :param eval_data:
    :param nodes:
    :return:
    """
    x, y, interpolations = view_transform(eval_data[0], nodes[1]-nodes[0],
                                          len(nodes), 0, nodes[-1])
    win = pg.plot(title="heavy chain visualization")
    param_plt = pg.PlotDataItem(symbol="o", symbolPen=pg.mkPen(None))
    scatter_plt = pg.ScatterPlotItem(pen=pg.mkPen(None), symbol="o")
    win.setXRange(np.min(x), np.max(x))
    win.setYRange(np.min(y), np.max(y))
    win.showGrid(x=True, y=True, alpha=.7)
    win.setAspectLocked(ratio=1, lock=True)
    win.addItem(param_plt)
    win.addItem(scatter_plt)
    param_plt.getViewBox().invertY(True)

    def update_plt():
        if "frame_cnt" not in update_plt.__dict__ or update_plt.frame_cnt == x.shape[0] - 1:
            update_plt.frame_cnt = 0
        else:
            update_plt.frame_cnt += 1
        param_plt.setData(x=x[update_plt.frame_cnt, :], y=y[update_plt.frame_cnt, :],
                          symbol=None, pen=pg.mkPen('b', width=2))

    hc_visualization.timer = pg.QtCore.QTimer()
    hc_visualization.timer.timeout.connect(update_plt)
    if show:
        hc_visualization.timer.start(1e3 * eval_data[0].input_data[0][1])

    return x, y


class ConstantInput(pi.SimulationInput):

    def __init__(self):
        pi.SimulationInput.__init__(self)

    def _calc_output(self, **kwargs):
        t = kwargs["time"]
        if t < 1:
            val = 0
        elif t < 2:
            val = -1
        elif t < 4:
            val = 0
        elif t < 5:
            val = 1
        else:
            val = 0

        return dict(output=val)


# spatial approximation order
N = 5

# temporal domain
T = 10
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 1.25
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.SimulationInputSum([
    ConstantInput()
    # pi.SignalGenerator('sawtooth', np.array(temp_dom), frequency=0.2,
    #                    scale=1, offset=0, phase_shift=0)
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

# system parameters
m_l = 1e0                 # [kg]
rho = 1.78# 1.78              # [kg/mm] -> line density
gravity = 9.81
A_hc = m_l/rho # 2.4*8*14.85 # m**2 -> cross sectional area of chain

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

# fem_base_integral = pi.LagrangeNthOrder.cure_interval(nodes, order=1)
# pi.visualize_functions(fem_base_integral.fractions[0], 100)
# funcs = fem_base_integral.fractions
# pi.LagrangeNthOrder.integrate(funcs, nodes)

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

# project on test functions
projections = list()
limits = (z, spat_bounds[0], spat_bounds[1])
tau = gravity * (A_hc * rho * l - A_hc * rho * z + m_l)
for psi_w, psi_v in zip(test_funcs_w, test_funcs_v):
    projections.append(
        sp.Integral(sp.diff(w_approx, t) * psi_w, limits)
        + sp.Integral(sp.diff(v_approx, t) * psi_v, limits)
        - sp.Integral(v_approx * psi_w, limits)
        + sp.Integral(sp.diff(w_approx, z) * psi_w, limits)
        - m_l * sp.diff(w_approx, z).subs(z, l_hc) * psi_w.subs(z, l_hc)
        + 1/(A_hc * rho) * u * psi_w.subs(z, 0)
        - sp.Integral(sp.diff(w_approx, z) * psi_w, limits)
        + (l_hc + m_l/(A_hc * rho)) * sp.Integral(sp.diff(w_approx, z) * sp.diff(psi_w, z), limits)
        - sp.Integral(sp.diff(w_approx, z) * sp.diff(psi_w, z) * z, limits)
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
win = pi.PgAnimatedPlot(data)

hc_visualization(data, nodes)

pi.show()
