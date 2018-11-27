import pyinduct as pi
import pyqtgraph as pg
import numpy as np

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
