r"""
This example considers a nonlinear tube reactor which can be described
with the normed variables/parameters:

- :math:`x_1(z,t)` ~ gas 1 concentration [mol/m3]

- :math:`x_2(z,t)` ~ gas 2 concentration [mol/m3]

- :math:`x_3(z,t)` ~ coverage rate [-]

- :math:`x_4(z,t)` ~ temperature [K]

- :math:`T_a(t)` ~ ambient temperature [K]

- :math:`v(t)` ~ velocity [m/s]

- :math:`u_1(t) = x_1(0,t)` ~ system input [mol/m3]

- :math:`u_2(t) = x_2(0,t)` ~ system input [mol/m3]

- :math:`u_3(t) = x_4(0,t)` ~ system input [K]

- :math:`k_1, k_2, k_3, k_4, \beta_1, \Omega, \alpha, E_1, E_2, E_3, E_4` ~ a bunch
  of parameters related to the chemical kinetics. The parameter values are copied
  from [SW16]_ .

by the following equations:

.. math::
    :nowrap:

    \begin{align*}
        \dot{x}_1(z,t) + v(t) x'_1 &=
                        - \Omega k_1 e^{-E_1 / R (1 / x_4(z,t) - 1 / x_{4,ref})} x_1(z,t)
                          (1 - x_3(z,t)) \\
                       &+ \Omega k_2 e^{-E_1 / R (1 / x_4(z,t) - 1 / x_{4,ref})} x_3(z,t) \\
        \dot{x}_2(z,t) + v(t) x'_2 &=
                          - \Omega k_2 e^{E_3 / R (1 / x_4(z,t) - 1 / x_{4,ref}) (1-\alpha x_3(z,t)} x_3(z,t) x_2(z,t) \\
        \dot{x}_3(z,t) &= k_1 e^{-E_1 / R (1 / x_4(z,t) - 1 / x_{4,ref})} x_1(z,t) (1 - x_3(z,t)) \\
                       &- k_2 e^{-E_2 / R (1 / x_4(z,t) - 1 / x_{4,ref}) (1-\alpha x_3(z,t)} x_3(z,t) \\
                       &- k_3 e^{-E_3 / R (1 / x_4(z,t) - 1 / x_{4,ref})} x_3(z,t) x_2(z,t) \\
                       &- k_4 e^{-E_4 / R (1 / x_4(z,t) - 1 / x_{4,ref})} x_3(z,t) \\
        \dot{x}_4(z,t) + v(t) x'_4  &= - \beta_1 (x_4(z,t) - T_a(t))
    \end{align*}


.. [SW16] Xander Seykens and Frank Willems;
          Modelling and control of diesel aftertreatment systems;
          IFAC AAC Pre-Symposium Tutorial, 2016, Norrköping
"""

# (sphinx directive) start actual script
from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import pickle
    from numbers import Number

    import sympy as sp
    import numpy as np
    import pyinduct as pi
    import pyinduct.sym_simulation as ss

    class SinusTrajectory(pi.SimulationInput):
        """
        Trajectory generator for a sinus as simulation input signal.

        Args:
            scale (numbers.Number): scale of sinus
            offset (numbers.Number): offset of sinus
            frequency (numbers.Number): frequency in Hz
            phase (numbers.Number): phase of sinus in rad
        """

        def __init__(self, scale=1, offset=0, frequency=1, phase=0, name=""):
            super().__init__(name)
            self._scale = scale
            self._offset = offset
            self._frequency = frequency
            self._phase = phase

        def _calc_output(self, **kwargs):
            if isinstance(kwargs["time"], (list, np.ndarray)):
                output = np.ones(len(np.atleast_1d(kwargs["time"]))) * self._startValue
                for idx, time in kwargs["time"]:
                    output[idx] = self._offset + self._scale * np.sin(
                        2 * np.pi * self._frequency * time
                        + self._phase)
                return dict(output=output)
            elif isinstance(kwargs["time"], Number):
                return dict(output=self._offset + self._scale * np.sin(
                    2 * np.pi * self._frequency * kwargs["time"] + self._phase))
            else:
                raise NotImplementedError

    class RampTrajectory(pi.SimulationInput):
        """
        Trajectory generator for a ramp as simulation input signal.

        Args:
            startValue (numbers.Number): Desired start value of the output.
            finalValue (numbers.Number): Desired value after step of the output
            stepStartTime (numbers.Number): Time where the ramp starts
            stepEndTime (numbers.Number): Time where the ramp ends
        """
        def __init__(self, startValue=0, finalValue=1, stepStartTime=1,
                     stepEndTime=2, name=""):
            super().__init__(name)
            self._startValue = startValue
            self._finalValue = finalValue
            self._stepStartTime = stepStartTime
            self._stepEndTime = stepEndTime
            self._m = (self._finalValue - self._startValue) / (
                self._stepEndTime - self._stepStartTime)
            self._n = self._startValue - self._m * self._stepStartTime

        def _calc_output(self, **kwargs):
            if isinstance(kwargs["time"], (list, np.ndarray)):
                output = np.ones(len(np.atleast_1d(kwargs["time"]))) * self._startValue
                for idx, time in kwargs["time"]:
                    if self._stepStartTime <= time <= self._stepEndTime:
                        output[idx] = self._m * time + self._n
                    elif time < self._stepStartTime:
                        output[idx] = self._startValue
                    else:
                        output[idx] = self._finalValue
                return dict(output=output)
            elif isinstance(kwargs["time"], Number):
                if self._stepStartTime <= kwargs["time"] <= self._stepEndTime:
                    return dict(output=self._m * kwargs["time"] + self._n)
                elif kwargs["time"] < self._stepStartTime:
                    return dict(output=self._startValue)
                else:
                    return dict(output=self._finalValue)
            else:
                raise NotImplementedError


    def initial_condition_x1(z):
        return 0

    def initial_condition_x2(z):
        return 0

    def initial_condition_x3(z):
        return 0

    def initial_condition_x4(z):
        return 250 + 273.15


    # define some bases
    domain = pi.Domain((0, 0.15), num=3)
    base = pi.LagrangeNthOrder.cure_interval(domain, order=1)
    pi.register_base("fem_base", base)
    # for i in range(1, 5):
    #     base = pi.LagrangeNthOrder.cure_interval(domain, order=1)
    #     pi.register_base("base_" + str(i), base)

    # parameters
    l = domain.bounds[1]
    k1 = 6
    E1 = 5e3
    k2 = 0.07
    E2 = 6.6e4
    k3 = 17.7
    E3 = 4.57e4
    k4 = 7e-3
    E4 = 1.50e5
    alpha = 0.4
    hga = 10
    Aga = 0.15 * 0.15 * 0.15
    cpGas = 1090
    rhoGas = 0.6
    beta1 = hga / (cpGas * rhoGas * Aga)
    R = 8.3145
    Omega = 102.8
    x4_ref = 600

    # interpolate = True

    # sympy symbols
    t = ss.time
    z = ss.space[0]

    u1, u2, u3 = [ss.get_input() for i in range(3)]
    x1, x2, x3, x4 = [ss.get_field_variable(z, t) for i in range(4)]

    ACat = 0.15 * 0.15
    eps = 0.64504929009858025
    v = (0.1 + 0.05 * sp.sin(2 * sp.pi * 1 / 10 * t)) / (ACat * eps)
    Ta = -7 * t + (80 + 273.15)

    input_dict = {
        u1: RampTrajectory(startValue=0,
                           finalValue=6.2e-2,
                           stepStartTime=0.5,
                           stepEndTime=1),
        u2: RampTrajectory(startValue=0,
                           finalValue=9e-2,
                           stepStartTime=3,
                           stepEndTime=4),
        u3: SinusTrajectory(scale=75,
                            offset=250 + 273.15,
                            frequency=1 / 10,
                            phase=1.5708)
    }

    # boundary conditions
    bc_x1 = sp.Eq(sp.Subs(x1.diff(z), z, 0), u1)
    bc_x2 = sp.Eq(sp.Subs(x2.diff(z), z, 0), u2)
    bc_x4 = sp.Eq(sp.Subs(x4.diff(z), z, 0), u3)

    x_1 = ss.create_approximation(z, "fem_base", [bc_x1])
    x_2 = ss.create_approximation(z, "fem_base", [bc_x2])
    x_3 = ss.create_approximation(z, "fem_base", [])
    x_4 = ss.create_approximation(z, "fem_base", [bc_x4])
    approximations = [x_1, x_2, x_3, x_4]

    psi1, psi2, psi3, psi4 = [ss.get_test_function(z) for i in range(4)]

    base_var_map = {
        x1: x_1,
        x2: x_2,
        x3: x_3,
        x4: x_4,
        psi1: x_1.base,
        psi2: x_2.base,
        psi3: x_3.base,
        psi4: x_4.base,
    }

    wf1 = (
        - ss.InnerProduct(x1.diff(t), psi1, domain.bounds)
        + v * (-(x1 * psi1).subs(z, l)
               + u1 * psi1.subs(z, 0)
               + ss.InnerProduct(x1, psi1.diff(z), domain.bounds)
               )
        - Omega * k1 * ss.InnerProduct(
              sp.exp(-E1 / R * (1 / x4 - 1 / x4_ref)) * x1 * (1 - x3),
              psi1,
        domain.bounds)
        + Omega * k2 * ss.InnerProduct(
              sp.exp(-E2 / R * (1 / x4 - 1 / x4_ref) * (1 - alpha * x3)) * x3,
              psi1,
        domain.bounds)
    )
    if 0:
        # weak formulations
        wf1 = WeakFormulation([
            IntegralTerm(Product(x_1.derive(temp_order=1), psi1),
                         limits=domain.bounds, scale=-1),
            SymbolicTerm(term=-v(t) * x1(l, t), test_function=psi1(l),
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * u1(t), test_function=psi1(0), input=input,
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * x1(z, t), test_function=psi1.derive(1),
                         base_var_map=base_var_map, input_var_map=input_var_map,
                         interpolate=interpolate),
            SymbolicTerm(term=(-Omega * k1 * sp.exp(-E1 / R * (1 / x4(z, t) - 1 / x4_ref)) * x1(z, t) *
                               (1 - x3(z, t)) +
                               Omega * k2 * sp.exp(-E2 / R *
                                                   (1 / x4(z, t) - 1 / x4_ref) *
                                                   (1 - alpha * x3(z, t))) * x3(z, t)),
                         test_function=psi1, base_var_map=base_var_map,
                         input_var_map=input_var_map, interpolate=interpolate)
        ], name="x_1")

    wf2 = (
        - ss.InnerProduct(x2.diff(t), psi2, domain.bounds)
        + v * (-(x2 * psi2).subs(z, l)
               + u2 * psi2.subs(z, 0)
               + ss.InnerProduct(x2, psi2.diff(z), domain.bounds)
               )
        - Omega * k3 * ss.InnerProduct(
            sp.exp(-E3 / R * (1 / x4 - 1 / x4_ref)) * x3 * x2,
            psi2,
    domain.bounds)
    )
    if 0:
        wf2 = WeakFormulation([
            IntegralTerm(Product(x_2.derive(temp_order=1), psi2),
                         limits=domain.bounds, scale=-1),
            SymbolicTerm(term=-v(t) * x2(l, t), test_function=psi2(l),
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * u2(t), test_function=psi2(0), input=input,
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * x2(z, t), test_function=psi2.derive(1),
                         base_var_map=base_var_map, input_var_map=input_var_map,
                         interpolate=interpolate),
            SymbolicTerm(term=-Omega * k3 * sp.exp(-E3 / R *
                                                   (1 / x4(z, t) - 1 / x4_ref)) * x3(z, t) * x2(z, t),
                         test_function=psi2, base_var_map=base_var_map,
                         input_var_map=input_var_map, interpolate=interpolate)
        ], name="x_2")

    wf3 = (
        - ss.InnerProduct(x3.diff(t), psi3, domain.bounds)
        + k1 * ss.InnerProduct(
        sp.exp(-E1 / R * (1 / x4 - 1 / x4_ref)) * x1 * (1 - x3),
        psi3,
        domain.bounds)
        - k2 * ss.InnerProduct(
        sp.exp(-E2 / R * (1 / x4 - 1 / x4_ref) * (1 - alpha * x3)) * x3,
        psi3,
        domain.bounds)
        - k3 * ss.InnerProduct(sp.exp(-E3 / R * (1 / x4 - 1 / x4_ref)) * x3 * x2,
                               psi3,
                               domain.bounds)
        - k4 * ss.InnerProduct(sp.exp(-E4 / R * (1 / x4 - 1 / x4_ref)) * x3,
                               psi3,
                               domain.bounds)
    )
    if 0:
        wf3 = WeakFormulation([
            IntegralTerm(Product(x_3.derive(temp_order=1), psi3),
                         limits=domain.bounds, scale=-1),
            SymbolicTerm(term=(k1 * sp.exp(-E1 / R * (1 / x4(z, t) - 1 / x4_ref)) * x1(z, t) * (1 - x3(z, t)) -
                               k2 * sp.exp(-E2 / R * (1 / x4(z, t) - 1 / x4_ref) * (1 - alpha * x3(z, t))) * x3(z, t) -
                               k3 * sp.exp(-E3 / R * (1 / x4(z, t) - 1 / x4_ref)) * x3(z, t) * x2(z, t) -
                               k4 * sp.exp(-E4 / R * (1 / x4(z, t) - 1 / x4_ref)) * x3(z, t)),
                         test_function=psi3, base_var_map=base_var_map,
                         input_var_map=input_var_map, interpolate=interpolate)
        ], name="x_3")

    wf4 = (
        - ss.InnerProduct(x4.diff(t), psi4, domain.bounds)
        + v * (-(x4 * psi4).subs(z, l)
               + u3 * psi4.subs(z, 0)
               + ss.InnerProduct(x4, psi4.diff(z), domain.bounds)
               )
        - beta1 * ss.InnerProduct(x4 - Ta, psi4, domain.bounds)
    )
    weak_forms = [wf1, wf2, wf3, wf4]

    if 0:
        wf4 = WeakFormulation([
            IntegralTerm(Product(x_4.derive(temp_order=1), psi4),
                         limits=domain.bounds, scale=-1),
            SymbolicTerm(term=-v(t) * x4(l, t), test_function=psi4(l),
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * u3(t), test_function=psi4(0), input=input,
                         base_var_map=base_var_map, input_var_map=input_var_map),
            SymbolicTerm(term=v(t) * x4(z, t), test_function=psi4.derive(1),
                         base_var_map=base_var_map, input_var_map=input_var_map,
                         interpolate=interpolate),
            SymbolicTerm(term=-beta1 * (x4(z, t) - Ta(t)), test_function=psi4,
                         base_var_map=base_var_map, input_var_map=input_var_map,
                         interpolate=interpolate),
            # dummy inputs
            ScalarTerm(Product(Input(u, index=0), psi4(0)), scale=0),
            ScalarTerm(Product(Input(u, index=1), psi4(0)), scale=0),
            ScalarTerm(Product(Input(u, index=2), psi4(0)), scale=0),
        ], name="x_4")
        weak_forms = [wf1, wf2, wf3, wf4]

    ic_dict = {
        x_1: initial_condition_x1,
        x_2: initial_condition_x2,
        x_3: initial_condition_x3,
        x_4: initial_condition_x4,
    }

    # now comes the interesting part
    rep_eqs = ss.substitute_approximations(weak_forms, base_var_map)
    sys, state, inputs = ss.create_first_order_system(rep_eqs)

    temp_domain = pi.Domain((0, 10), num=101)
    np.seterr(under="warn", over="warn")
    res = ss.simulate_state_space(temp_domain, sys, ic_dict, input_dict,
                                  inputs, state)

    # post processing
    t_dom = pi.Domain(points=res_weights.t)
    weight_dict = ss._sort_weights(res_weights.y, state, [approximations])
    results = ss._evaluate_approximations(weight_dict,
                                          [approximations],
                                          temp_domain,
                                          domain)

    # visualization
    win = pi.PgAnimatedPlot(results)
    wins = pi.PgSurfacePlot(results[0])
    pi.show()
