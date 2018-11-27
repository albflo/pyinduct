import pyinduct as pi
import numpy as np
import sympy as sp

__all__ = ["IntegrateFunction", "ConstantInput"]


class IntegrateFunction(pi.Function):
    """
    Integrate Function to extend pyinduct function with integration

    The class gets initialized by the testfunction of choice and the limits of
    the integration. Afterwards the class method __call__ can be called to
    return the numerical value of the integration of the desired function.
    """
    def __init__(self, eval_handle, limits_int, domain=(-np.inf, np.inf),
                 nonzero=(-np.inf, np.inf), derivative_handles=None):
        if not isinstance(eval_handle, pi.Function):
            raise TypeError("Eval_handle is not a pi.Function!")
        self.func = eval_handle
        if (isinstance(limits_int[0], sp.Symbol) and
            isinstance(limits_int[1], sp.Symbol)):
            raise TypeError("Both limits of integral are symbols!")
        elif isinstance(limits_int[0], sp.Symbol):
            self.lowerlimit = True
            self.numerical = False
            self.limit = limits_int[1]
        elif isinstance(limits_int[1], sp.Symbol):
            self.lowerlimit = False
            self.numerical = False
            self.limit = limits_int[0]
        else:
            self.numerical = True
            self.limits = limits_int
        super(IntegrateFunction, self).__init__(eval_handle, domain, nonzero,
                                                derivative_handles)

    def __call__(self, z=0):
        if self.numerical:
            limits = {self.limits}
        elif self.lowerlimit:
            limits = {(z, self.limit)}
        else:
            limits = {(self.limit, z)}
        return pi.integrate_function(self.func, limits)[0].item(0)


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
