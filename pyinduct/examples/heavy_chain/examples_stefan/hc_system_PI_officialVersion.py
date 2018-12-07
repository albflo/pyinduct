# Changed only the pyinduct expressions so that stefans simulation runs with the
# current version. The results of the calculation are the exact same!
import numpy as np
from scipy import interpolate
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
import pyinduct as pi

class ConstantInput(pi.SimulationInput):

    def __init__(self):
        pi.SimulationInput.__init__(self)

    def _calc_output(self, **kwargs):
        t = kwargs["time"]
        if t < 1:
            val = 0
        elif t < 2:
            val = force
        elif t < 4:
            val = 0
        elif t < 5:
            val = -force
        else:
            val = 0

        return dict(output=val)


def heavy_chain(func_label, input_handle, l=1, ml=1.0, g=9.81, rho=1.0):
    """
    returns weak formulation of the heavy chain
    :param shape_funcs:
    :param input_handle:
    :param l:
    :param ml:
    :param g:
    :param rho:
    """
    interval = (0, l)
    alpha_0 = 1
    alpha_1 = alpha_0 * ml / rho

    w = pi.FieldVariable(func_label)
    phi = pi.TestFunction(func_label)

    dt_terms = [
        pi.IntegralTerm(pi.Product(
            w.derive(temp_order=2),
            phi), interval, scale=alpha_0
        ),
        pi.ScalarTerm(pi.Product(
            w.derive(temp_order=2)(0),
            phi(0)
            ), scale=alpha_1
        )
    ]

    tau = pi.Function(lambda z: g*(ml + rho*z))
    pi.register_base("tau", pi.Base([tau]))
    a_terms = [
        pi.IntegralTerm(
            pi.Product(
                pi.Product(
                    w.derive(spat_order=1),
                    pi.ScalarFunction("tau")
                ),
                phi.derive(1)
            ), interval, scale=alpha_0/rho
        ),
        pi.ScalarTerm(
            pi.Product(
                pi.Input(input_handle),
                phi(l)
            ), scale=-alpha_0/rho
        )

    ]
    return pi.WeakFormulation(dt_terms + a_terms, name="hc")


def view_transform(data, node_dist, node_cnt, x_offset, y_offset):
    """
    transforms simulated data into from that is better to view
    :param data: eval_data from simulation run
    :return: transformed data
    """
    print("preparing visualization data ...")
    t_values = data.input_data[0]
    t_step = t_values[1]
    z_values = data.input_data[1]
    z_step = z_values[1]
    # invert spatially since iteration is easier from 0 to l
    w_values = data.output_data[..., ::-1]

    if True:
        # gradient method
        w = w_values
        grad = np.gradient(w, t_step, z_step)
        w_dt = grad[0]
        w_dz = grad[1]
    else:
        # helper funcs
        def find_nearest(array, values):
            vals = np.zeros(values.shape)
            for idx, val in enumerate(values):
                vals[idx] = find_nearest_1d(array, val)
            return vals

        def find_nearest_1d(array, value):
            idx = (np.abs(array-value)).argmin()
            return array[idx]

        # perform bivariate spline interpolation
        interp_points = np.array([(.5 + node_idx)*node_dist for node_idx in
                                                            range(node_cnt)])
        avail_interp_points = find_nearest(z_values, interp_points)
        interp_idx = np.in1d(z_values, avail_interp_points)
        red_z_data = z_values[interp_idx]
        red_output_data = w_values[:, interp_idx]
        xg, yg = np.meshgrid(t_values, red_z_data, indexing="ij")

        if True:
            # variant a
            inter_spline = interpolate.SmoothBivariateSpline(xg.flatten(),
                                        yg.flatten(), red_output_data.flatten(),
                                        kx=2, ky=2, s=0)
            w = inter_spline(t_values, z_values)
            w_dt = inter_spline(t_values, z_values, dx=1)
            w_dz = inter_spline(t_values, z_values, dy=1)
        else:
            # variant b
            tck = interpolate.bisplrep(xg, yg, red_output_data, s=10)
            w = interpolate.bisplev(data.input_data[0], data.input_data[1], tck)
            w_dt = interpolate.bisplev(data.input_data[0], data.input_data[1],
                                       tck, 1, 0)
            w_dz = interpolate.bisplev(data.input_data[0], data.input_data[1],
                                       tck, 0, 1)

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

    print("done.")
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
    win = pg.plot(title="heavy chain")
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

        if 0:
            # TODO add nice circles for masses
            x_koords = [x[update_plt.frame_cnt, 0], x[update_plt.frame_cnt, -1]]
            y_koords = [chain_length, y[update_plt.frame_cnt, -1]]
            scatter_plt.setData(x=x_koords, y=y_koords,
                                pen=pg.mkPen(None),
                                symbolSize=[wagon_mass, load_mass],
                                pxMode=[True, True],
                                symbol=["o", "o"],
                                symbolPen=[pg.mkPen(None), pg.mkPen(None)],
                                symbolBrush=[pg.mkBrush(255, 255, 255, 255),
                                             pg.mkBrush(255, 255, 255, 255)])

    hc_visualization.timer = pg.QtCore.QTimer()
    hc_visualization.timer.timeout.connect(update_plt)
    if show:
        hc_visualization.timer.start(1e3 * eval_data[0].input_data[0][1])

    return x, y

app = QApplication([])

N = 20

load_mass = 1e0
a_roh = 1e2
gravity = 9.81
chain_length = 1
force = 1e1

z_start = 0
z_end = chain_length
z_step = .5
spat_dom = pi.Domain(bounds=(z_start, z_end), step=z_step)

t_start = 0
t_end = 10
t_step = 0.01
temp_dom = pi.Domain(bounds=(t_start, t_end), step=t_step)

node_distance = chain_length / N

# initial conditions
def w_func(z, t):
    """
    initial conditions for testing
    """
    # return np.sin(z) + t
    return 0


def w_dt_func(z , t):
    """
    initial conditions for testing
    """
    # return np.sin(z) + t
    # return 1*z
    return 0


ic = np.array([
    pi.Function(lambda z: w_func(z, 0)),
    pi.Function(lambda z: w_dt_func(z, 0)),
])

# pde
nodes = pi.Domain(spat_dom.bounds, num=N)
init_funcs = pi.LagrangeFirstOrder.cure_interval(nodes)
func_label = 'init_funcs'
pi.register_base(func_label, init_funcs)
u = ConstantInput()
chain_pde = heavy_chain(func_label, input_handle=u, l=chain_length,
                        ml=load_mass, g=gravity, rho=a_roh)

# simulate system
eval_data = pi.simulate_system(chain_pde, ic, temp_dom, spat_dom)

# display results
# pt = pi.PgAnimatedPlot(eval_data)
# sp_modal = pi.PgSurfacePlot(eval_data[0])

# visualize
# hc_visualization(eval_data, nodes)

# run main event loop
# app.exec_()

del app
