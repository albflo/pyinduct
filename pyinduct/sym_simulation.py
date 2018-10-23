"""
New simulation module approach that makes use of sympy for expression handling
"""

from time import clock
from copy import copy
import warnings
import sympy as sp
from sympy.utilities.autowrap import autowrap
# import symengine as sp
from sympy.utilities.autowrap import ufuncify
import numpy as np
from scipy.integrate import solve_ivp, ode
from tqdm import tqdm
from jitcode import jitcode, t, y, UnsuccessfulIntegration

from matplotlib import pyplot as plt

from .registry import get_base
from .core import (Domain, Function, Base,
                   domain_simplification, domain_intersection,
                   integrate_function,
                   project_on_base, EvalData, Bunch)
from .simulation import simulate_state_space as old_ss_sim

time = sp.symbols("t", real=True)
space = sp.symbols("z:3", real=True)


# parameter database
_parameters = {}


def register_parameters(*args, **kwargs):
    if args:
        new_params = dict(args)
        _parameters.update(new_params)

    if kwargs:
        _parameters.update(kwargs)


def get_parameter(param):
    return _parameters.get(param, None)


def get_parameters(*args):
    if not args:
        # return _parameters.items()
        return _parameters
    else:
        return [_parameters.get(arg, None) for arg in args]


# fraction database
_fraction_map = {}


def get_base_fraction_symbol(frac, *syms):
    global _fraction_map
    if not isinstance(frac, Function):
        return frac

    if frac in _fraction_map:
        return _fraction_map[frac]

    func = get_function(*syms)
    func.func._imp_ = staticmethod(frac)
    _fraction_map.update({frac: func})

    return func


_weight_cnt = 0
_weight_letter = "_c"


def get_weight():
    global _weight_cnt
    w = sp.Function("{}_{}".format(_weight_letter, _weight_cnt),
                    real=True)(time)
    _weight_cnt += 1
    return w


def get_weights(num):
    if num < 0:
        raise ValueError
    c = [get_weight() for i in range(num)]
    return c


_function_cnt = 0
_function_letter = "_f"


def get_function(sym):
    global _function_cnt
    f = sp.Function("{}_{}".format(_function_letter, _function_cnt),
                    real=True)(sym)
    _function_cnt += 1
    return f


_field_var_cnt = 0
_field_var_letter = "_x"


def get_field_variable(*symbols):
    global _field_var_cnt
    x = sp.Function("{}_{}".format(_field_var_letter, _field_var_cnt),
                    real=True)(*symbols)
    _field_var_cnt += 1
    return x


_test_function_cnt = 0
_test_function_letter = "_g"


def get_test_function(*symbols):
    global _test_function_cnt
    g = sp.Function("{}_{}".format(_test_function_letter, _test_function_cnt),
                    real=True)(*symbols)
    _test_function_cnt += 1
    return g


_input_cnt = 0
_input_letter = "_u"


def get_input(*symbols):
    if not symbols:
        symbols = [time]
    global _input_cnt
    u = sp.Function("{}_{}".format(_input_letter, _input_cnt),
                    real=True)(*symbols)
    _input_cnt += 1
    return u


_lambda_cnt = 0
_lambda_letter = "_l"


def get_lambda(*args):
    global _lambda_cnt
    l = sp.Function("{}_{}".format(_lambda_letter, _lambda_cnt),
                    real=True)(*args)
    _lambda_cnt += 1
    return l


def build_lag1st(sym, start, mid, end):
    if start == mid:
        obj = sp.Piecewise((0, sym < mid),
                           (1 - (sym - mid)/(end - mid), sym < end),
                           (0, sym >= end)
                           )
    elif mid == end:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (0, sym > mid)
                           )
    else:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym < mid),
                           (1, sym == mid),
                           (1 - (sym - mid)/(end - mid), sym < end),
                           (0, sym >= end)
                           )
    return obj


def create_lag1st_base(sym, bounds, num):
    # define base vectors
    nodes = Domain(bounds, num)
    funcs = [build_lag1st(sym, *nodes[[0, 0, 1]])]
    for idx in range(1, num-1):
        funcs.append(build_lag1st(sym, *nodes[[idx-1, idx, idx+1]]))
    funcs += [build_lag1st(sym, *nodes[[-2, -1, -1]])]

    if 0:
        # plotting of Piecewise fails really bad, so lambdify first
        fig = plt.figure()
        vals = np.linspace(0, 1, num=1e3)
        for f in funcs:
            fn = sp.lambdify(z, f, modules="numpy")
            res = fn(vals)
            plt.plot(vals, res)
        plt.show()

    return funcs


class LumpedApproximation:

    def __init__(self, symbols, ess_expr, nat_expr, weights, base_map, bcs):
        self._syms = symbols
        self._ess_expr = ess_expr
        self._nat_expr = nat_expr
        self._weights = weights
        self._base_map = base_map
        self._bcs = bcs

        # check for extra dependencies
        self._extra_args = list(_find_inputs([self.expression]))

        # substitute all known functions and symbols and generate callback
        self._cb = sp.lambdify(self._syms + self.weights + self._extra_args,
                               self.expression.subs(get_parameters()),
                               modules="numpy")

    @property
    def expression(self):
        return self._ess_expr + self._nat_expr

    @property
    def weights(self):
        return self._weights

    @property
    def extra_args(self):
        return self._extra_args

    @property
    def base_map(self):
        return self._base_map

    @property
    def base(self):
        """ Return all base fractions that do not coincide with essential bcs
        """
        return [frac for frac, ess in self._base_map.values() if not ess]

    @property
    def bcs(self):
        return self._bcs

    def __call__(self, *args, **kwargs):
        if args:
            return self._cb(*args)
        if kwargs:
            return self._cb(*[kwargs[w] for w in self._weights])

    def approximate_function(self, func, *extra_args, use_collocation=False):
        """
        Project the given function into the subspace of this Approximation

        Args:
            func(sp.Function): Function expression to approximate
            extra_args(list): Extra arguments needed for evaluation.
        """
        f = sp.sympify(func)
        hom_func = (f - self._ess_expr).subs(get_parameters())
        hom_func = hom_func.subs(list(zip(self._extra_args, extra_args)))

        # if use_collocation:
        #     weights = []
        #     for f in self.base:
        #         w = f.evalf()

        cb = sp.lambdify(self._syms, hom_func, modules="numpy")

        weights = project_on_base(Function(cb), Base(self.base))
        return weights

    def get_spatial_approx(self, weights, *extra_args):
        def spat_eval(*syms):
            return self._cb(syms, weights, extra_args)

        # return spat_eval
        expr = self.expression.subs(get_parameters())
        expr = expr.subs(list(zip(self.weights, weights))
                         + list(zip(self.extra_args, extra_args)))
        return sp.lambdify(self._syms, expr, modules="numpy")

    # def subs(self, *args):
    #     new_symbols = [sym.subs(*args) for sym in self._syms]
    #     new_ess_expr = self._ess_expr.subs(*args)
    #     new_nat_expr = self._nat_expr.subs(*args)
    #     new_bcs = [bc.subs(*args) for bc in self._bcs]
    #
    #     new_obj = LumpedApproximation(self._syms,
    #                                   new_ess_expr,
    #                                   new_nat_expr,
    #                                   self._weights,
    #                                   self._base_map,
    #                                   new_bcs)
    #
    #     return new_obj


class InnerProduct(sp.Expr):
    """
    An Inner Product on L2

    Args:
        a: Left Operand .
        b: Right Operand .

    """
    is_complex = True

    def __new__(cls, left, right, bounds):
        # identify free variables of arguments
        variables = left.atoms(sp.Symbol).union(right.atoms(sp.Symbol))

        # extract spatial coordinates
        variables = variables.intersection(space)

        # construct integrals over remaining coordinates
        # TODO .coeff seems to have problems with conjugate
        kernel = left * right
        # kernel = sp.conjugate(left) * right
        for dim in variables:
            xab = (dim, bounds[0], bounds[1])
            kernel = FakeIntegral(kernel, xab)
        return kernel

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    def doit(self, **hints):
        pass


class FakeIntegral(sp.Integral):
    """
    Placeholder class that looks like an Integral

    The difference is that this class but won't try to integrate anything
    if not directly told to do so. Using this approach, integral expression
    can be handled and simplified more easily, until eventually they are
    solved numerically.
    """

    def doit(self, **hints):
        """
        Do not try to integrate anything, just perform operations on the args
        """
        return self.func(*[arg.doit() for arg in self.args])

    def eval_numerically(self):
        """
        Build function for numerical evaluation of this expression

        If the function only depends on known symbols, calculate its value.

        Returns:
            Either the result value or a Function object with `_imp_` attribute
            to calculate the result during runtime.
        """
        kernel, (sym, a, b) = self.args

        # Identify time dependent terms the integral might depend on
        eval_args = []
        if time in self.free_symbols:
            eval_args.append(time)
        eval_args += list(_find_weights(self.args))

        # build callback for spatial integration
        f = sp.lambdify([sym] + eval_args, kernel, modules="numpy")

        # extract domains
        domain = {(-np.inf, np.inf)}
        for func in kernel.atoms(sp.Function):
            if hasattr(func, "_imp_"):
                new_dom = func._imp_.domain
                domain = domain_intersection(domain, new_dom)

        nonzero = self.get_nonzero_area(kernel)
        kernel_domain = domain_intersection(domain, nonzero)

        # kernel is zero or undefined on the whole region of the integral
        if not kernel_domain:
            return 0

        # build dummy function that will compute the integral
        def _eval_integral(*args):
            up_a, up_b = [lim.subs(zip(eval_args, args)) for lim in [a, b]]
            integral_domain = {(float(up_a), float(up_b))}
            interval = domain_intersection(kernel_domain, integral_domain)
            if 0:
                reg = next(iter(interval))
                x_vals = np.linspace(float(a), float(b))
                plt.plot(x_vals, f(x_vals).T)
                plt.axhline(xmin=reg[0], xmax=reg[1], c="r")
                plt.show()
            res, err = integrate_function(f, interval, args=args)
            return res

        if not eval_args:
            return _eval_integral()

        warnings.warn("Creating integral callback for evaluation at runtime")

        int_func = get_lambda(*eval_args)
        int_func.func._imp_ = staticmethod(_eval_integral)
        return int_func

    def get_nonzero_area(self, kernel):
        """
        Identify nonzero area by propagating nonzero areas through the expression

        For example:
            k=f1(z) * f2(z) -> N(k) = N(f1) ^ N(f2)
            k=f1(z) + f2(z) -> N(k) = N(f1) v N(f2)
        or more general:
            k=g(f1(z)) -> N(k) = N(g(f1))
        """
        if isinstance(kernel, sp.Function):
            if hasattr(kernel.func, "_imp_"):
                n_area = kernel.func._imp_.nonzero
                return n_area

        if not kernel.args:
            # end of expr tree
            return {(-np.inf, np.inf)}

        areas = []
        for arg in kernel.args:
            area = self.get_nonzero_area(arg)
            areas.append(area)

        if isinstance(kernel, sp.Add):
            return domain_simplification(set.union(*areas))
        elif isinstance(kernel, sp.Mul):
            res = {(-np.inf, np.inf)}
            for a in areas:
                res = domain_intersection(res, a)
            return res
        elif isinstance(kernel, (sp.Pow, sp.Function)):
            # propagate the argument's area
            return areas[0]
        else:
            raise NotImplementedError(kernel)


class Lagrange1stOrder(sp.Piecewise):

    def __new__(cls, start, mid, end):
        x = sp.Symbol("x")
        obj = sp.Piecewise.__new__(cls, (0, x < start),
                                   ((x-start)/(mid - start), x < mid),
                                   (1, x == mid),
                                   (1 - (x-mid)/(end - mid), x < end),
                                   (0, x >= end),
                                   )
        return obj


def build_lag1st(sym, start, mid, end):
    if start == mid:
        obj = sp.Piecewise((0, sym < mid),
                           (1 - (sym - mid)/(end - mid), sym <= end),
                           (0, sym > end)
                           )
    elif mid == end:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (0, sym > mid)
                           )
    else:
        obj = sp.Piecewise((0, sym < start),
                           ((sym - start)/(mid - start), sym <= mid),
                           (1 - (sym - mid)/(end - mid), sym <= end),
                           (0, sym > end)
                           )
    return obj


def create_lag1ast_base(sym, bounds, num):
    # define base vectors
    nodes = Domain(bounds, num)
    funcs = [build_lag1st(sym, *nodes[[0, 0, 1]])]
    for idx in range(1, num-1):
        funcs.append(build_lag1st(sym, *nodes[[idx-1, idx, idx+1]]))
    funcs += [build_lag1st(sym, *nodes[[-2, -1, -1]])]

    if 0:
        # plotting of Piecewise fails really bad, so lambdify first
        fig = plt.figure()
        vals = np.linspace(0, 1, num=1e3)
        for f in funcs:
            fn = sp.lambdify(z, f, modules="numpy")
            res = fn(vals)
            plt.plot(vals, res)
        plt.show()

    return funcs


def create_approximation(syms, base_lbl, boundary_conditions, weights=None):
    """
    Create a lumped approximation of a distributed variable

    The approximation is performed as a product-ansatz using the functions
    registered under *base_lbl*. If essential boundaries are detected, they
    are homogenized, leading to a possible reduction of free weights in the
    resulting approximation.

    Args:
        syms((list of) Symbol(s)): Symbol(s) to use for approximation.
            (Dimension to discretise)
        base_lbl(str): Label of the base to use
        boundary_conditions(list): List of boundary conditions that have to be
            met by the approximation.

    Keyword Args:
        weights(list): If provided, weight symbols to use for the approximation.
            By default (None), a new set of weights will be created.

    Returns:
        LumpedApproximation: Object holding all information about the
        approximation.
    """
    if isinstance(syms, sp.Basic):
        syms = [syms]

    if len(syms) > 1:
        raise NotImplementedError("Higher dimensional cases are still missing")
    sym = syms[0]
    base = get_base(base_lbl)
    ess_bcs, nat_bcs, boundary_positions = _classify_boundaries(
        sym,
        boundary_conditions)

    ess_pairs = []
    ess_idxs = []
    # find the corresponding functions for the essential boundaries
    for cond, pos in ess_bcs:
        # identify all base fractions that differ from zero
        for idx, func in enumerate(base):
            if isinstance(func, Function):
                res = func(pos)
            elif isinstance(func, sp.Basic):
                res = func.subs(sym, pos)
            else:
                raise NotImplementedError

            if not np.isclose(float(res), 0):
                ess_pairs.append((cond, func))
                ess_idxs.append(idx)

    if len(ess_pairs) < len(ess_bcs):
        # no suitable base fraction for homogenisation found, create one
        inhom_pos = pos
        hom_pos = next((p for p in boundary_positions if p != pos))
        hom_frac = create_hom_func(sym, inhom_pos, hom_pos)
        ess_pairs.append((cond, hom_frac))

    assert len(ess_pairs) == len(ess_bcs)

    base_mapping = {}

    # inhomogeneous parts that are to be excluded
    x_ess = 0
    for cond, func in ess_pairs:
        d_func = get_base_fraction_symbol(func, sym)
        x_ess += cond[1] * d_func
        base_mapping[d_func] = func, True

    # extract shape functions which are to be kept
    nat_funcs = [func for idx, func in enumerate(base) if idx not in ess_idxs]

    # homogeneous part
    x_nat = 0
    if weights is None:
        weights = get_weights(len(nat_funcs))
    for idx, func in enumerate(nat_funcs):
        d_func = get_base_fraction_symbol(func, sym)
        x_nat += weights[idx] * d_func
        base_mapping[d_func] = func, False

    x_approx = LumpedApproximation([sym],
                                   x_ess, x_nat,
                                   weights,
                                   base_mapping,
                                   boundary_conditions)

    return x_approx


def _classify_boundaries(sym, boundary_conditions):
    """
    Identify essential and natural boundaries

    Since Dirichlet boundaries have to be handled different, this function takes
    a list of Equation objects and classifies their type into essential
    (dirichlet) and natural (neumann or robin) boundary conditions.

    Note:
        For the detection to work, substitutions must not have been performed.
        Therefore, create expression like `Subs(x(z,t), z, 0)` in favour of
        `x(z,t).subs(z, 0)`.
        However, using symengine this will also fail.

    Args:
        sym(Symbol): Symbol, specifying the coordinate to classify for.
        boundary_conditions(list): List of equations, defining the boundaries.

    Returns:
        3-tuple of essential boundaries, natural boundaries and all boundary
        positions.
    """
    # TODO: Pass placeholder expr and directly search for that
    boundary_positions = []
    ess_bcs = []
    nat_bcs = []
    for cond in boundary_conditions:
        lhs_idx = next((idx for idx, arg in enumerate(cond.args)
                        if _field_var_letter in str(arg.atoms(sp.Function))))
        lhs = cond.args[lhs_idx]
        rhs = cond.args[1 - lhs_idx]
        if isinstance(lhs, sp.Subs):
            assert sym in lhs.args[1]
            assert len(lhs.args[2]) == 1
            pos = next(iter(lhs.args[2]))
            expr = lhs.args[0]
        else:
            # go f*ck yourself symengine
            raise NotImplementedError

        pair = ((lhs, rhs), pos)

        if isinstance(expr, sp.Derivative):
            nat_bcs.append(pair)
        else:
            ess_bcs.append(pair)

        boundary_positions.append(pos)

    return ess_bcs, nat_bcs, boundary_positions


def create_hom_func(sym, pos_a, pos_b, mode="linear"):
    r"""
    Build a symbolic expression for homogenisation purposes.

    This function will evaluate to one at *pos_a* and to zero at *pos_b*.

    Args:
        sym: Symbol to use.
        pos_a: First position.
        pos_b: Second Position.

    Keyword Args:
        mode(str): Function type to use, choices are "linear", "trig" and "exp".

    Returns: Sympy expression

    """
    if mode == "linear":
        variables = sp.symbols("_m _n")
        _f = variables[0]*sym + variables[1]
    elif mode == "trig":
        variables = sp.symbols("_a _b")
        _f = sp.cos(variables[0] * sym + variables[1])
    # elif mode == "exp":
    #     variables = sp.symbols("_c _b")
    #     _f = sp.cos(variables[0]*sym + variables[1])
    else:
        raise NotImplementedError

    eqs = [_f.subs(pos_a) - 1, _f.subs(pos_b - 0)]
    sol = sp.solve(eqs, variables, dict=True)[0]
    res = _f.subs(sol)

    return res


def substitute_approximations(weak_form, mapping):
    """
    Substitute placeholder functions with their approximations

    Firstly, all testfunctions are substituted, by adding a new equation for
    every provided entity. Afterwards, the approximations for all other symbols
    are performed.

    Args:
        weak_form (list): List of sympy expression, representing the weak
            formulation of a system.
        mapping (dict): Dict holding the substitution mappings

    Returns:
        Extended equation system with substituted mapping.
    """

    print(">>> expanding generic formulation")
    ext_form, ext_mapping = _expand_kth_terms(weak_form, mapping)

    def wrap_expr(expr, *sym):
        def wrapped_func(*_sym):
            return expr.subs(list(zip(sym, _sym)))

        return wrapped_func

    def gen_func_subs_pair(func, expr):
        if isinstance(func, sp.Function):
            a = func.func
            args = func.args
        elif isinstance(func, sp.Subs):
            a = func
            if isinstance(func.args[0], sp.Derivative):
                args = func.args[0].args[0].args
            elif isinstance(func.args[0], sp.Function):
                args = func.args[0].args
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        b = wrap_expr(expr, *args)
        return a, b

    # create substitution lists
    rep_list = []
    approximations = []
    for sym, approx in ext_mapping.items():
        if isinstance(approx, LumpedApproximation):
            rep_list += [gen_func_subs_pair(*bc.args) for bc in approx.bcs]
            rep_list.append(gen_func_subs_pair(sym, approx.expression))
            approximations.append(approx)
        else:
            rep_list.append(gen_func_subs_pair(sym, approx))

    print(">>> substituting symbolic approximations")
    rep_eqs = []
    for eq in tqdm(ext_form):
        rep_eq = eq
        for pair in rep_list:
            rep_eq = rep_eq.replace(*pair, simultaneous=True)

        rep_eqs.append(rep_eq.doit())

    return sp.Matrix(rep_eqs), approximations


def _expand_kth_terms(weak_form, mapping):
    """
    Search for kth terms and expand the equation system by their mappings

    For now, a *k-th term* is recognized by its placeholder function being
    a test function. Since in the weak form, it acts as a placeholder for all
    test functions, an equation is added to the equation system for every entry
    in the value, belonging to a test-function key.

    Therefore, having one equation with two *k-th terms* and a mapping assigning
    10 function to each of them, yo will end up with 100 equations.
    """
    new_exprs = []
    new_mapping = {}
    kth_placeholders = {}

    # search for kth placeholders
    for sym, approx in mapping.items():
        if _test_function_letter in str(sym):
            kth_placeholders[sym] = approx
        else:
            new_mapping[sym] = approx

    # replace the kth occurrences by adding extra equations for every entry
    for eq in weak_form:
        occurrences = [sym in eq.atoms(sp.Function)
                       for sym in kth_placeholders.keys()]
        if not any(occurrences):
            # if no placeholder is used, directly forward the equation
            new_exprs.append(eq)
        else:
            for sym, approx in kth_placeholders.items():
                new_eqs, new_map = _substitute_kth_occurrence(eq, sym, approx)
                new_exprs += new_eqs
                new_mapping.update(new_map)

    return sp.Matrix(new_exprs), new_mapping


def _substitute_kth_occurrence(equation, symbol, expressions):
    new_eqs = []
    mappings = {}
    for expr in expressions:
        if equation.atoms(symbol):
            _g = get_base_fraction_symbol(expr, *symbol.args)
            if not isinstance(expr, Function):
                _g = get_test_function(*symbol.args)
                mappings[_g] = expr

            r_tpl = symbol.func, _g.func
            new_eqs.append(equation.replace(*r_tpl))

    return new_eqs, mappings


def create_first_order_system(weak_forms, input_map):

    new_forms = sp.Matrix(weak_forms)

    print(">>> simplifying integrals")
    new_forms = _simplify_integrals(new_forms)
    # new_forms = weak_forms
    # sp.pprint(new_forms, num_columns=200)

    print(">>> substituting derivatives")
    new_forms, targets = _convert_derivatives(new_forms)
    # sp.pprint(new_forms, num_columns=200)

    print(">>> substituting parameters")
    new_forms = new_forms.xreplace(get_parameters())
    # sp.pprint(new_forms, num_columns=200)

    print(">>> solving integrals")
    new_forms = _solve_integrals(new_forms)
    # sp.pprint(new_forms, num_columns=200)

    print(">>> running remaining evaluations")
    new_forms = _run_evaluations(new_forms)
    # sp.pprint(ss_form)

    print(">>> identifying inputs")
    inputs = _find_inputs(new_forms)
    sorted_inputs = sorted(inputs, key=lambda x: str(x))
    # sp.pprint(sorted_inputs)

    print(">>> identifying state components")
    state_elems = _find_weights(new_forms)
    sorted_state = sp.Matrix(sorted(state_elems,
                                    key=lambda x: float(str(x)[3:-3])))
    # sp.pprint(sorted_state)

    print(">>> solving for targets:")
    if 0:
        sol_dict = sp.solve(new_forms, targets)
        # sp.pprint(ss_form)

        print(">>> building statespace form")
        ss_form = sp.Matrix([sol_dict[s.diff(time)] for s in sorted_state])
        # sp.pprint(ss_form)
    else:
        sorted_targets = [s.diff(time) for s in sorted_state]
        # rep_map = {t: d for t, d in zip(sorted_targets, dummy_targets)}
        print("\t-collecting coefficient matrices")
        A, b = sp.linear_eq_to_matrix(sp.Matrix(new_forms), *sorted_targets)
        if 0:
            sp.pprint(A, num_columns=200)
            a, c = sp.linear_eq_to_matrix(b, *sorted_state)
            sp.pprint(a, num_columns=200)
            sp.pprint(c, num_columns=200)
            quit()

        print("\t-solving equation system")
        ss_form = A.LUsolve(b)

    input_derivatives = [d for d in _find_derivatives(weak_forms, time)
                         if d.args[0] in inputs]
    if input_derivatives:
        print(">>> eliminating input derivatives")
        ss_form, new_state, state_trafos = _eliminate_input_derivatives(
            input_derivatives,
            ss_form,
            sorted_state,
            inputs)
    else:
        state_trafos = tuple()

    ss_sys = Bunch(rhs=ss_form,
                   state=sorted_state,
                   inputs=sorted_inputs,
                   input_map=input_map,
                   orig_state=sorted_state,
                   transformations=state_trafos)

    return ss_sys


def _run_evaluations(weak_forms):
    new_forms = weak_forms.doit()

    funcs = new_forms.atoms(sp.Function)
    val_mapping = {}
    for f in funcs:
        # if all dependencies are satisfied, evaluate
        if not f.free_symbols:
            res = f.evalf()
            val_mapping.update({f: res})

    new_forms = new_forms.xreplace(val_mapping)
    return new_forms


def simulate_system(weak_forms, approx_map, input_map, ics, temp_dom, spat_dom):

    # build complete form
    rep_eqs, approximations = substitute_approximations(weak_forms, approx_map)

    # convert to state space system
    ss_sys = create_first_order_system(rep_eqs, input_map)

    # process initial conditions
    y0 = calc_initial_sate(ss_sys, ics, temp_dom[0])

    # simulate
    t_dom, sim_state = simulate_state_space(ss_sys, y0, temp_dom)

    # extract original state
    state_traj, input_traj = calc_original_state(sim_state, ss_sys, t_dom)

    results = process_results(approximations, input_traj, spat_dom, ss_sys,
                              state_traj, t_dom)

    return results


def process_results(approximations, input_traj, spat_dom, ss_sys, state_traj,
                    t_dom):
    print(">>> processing simulation results")
    weight_dict, extra_dict = _sort_weights(state_traj, input_traj, ss_sys,
                                            approximations)
    results = _evaluate_approximations(weight_dict, extra_dict, approximations,
                                       t_dom, spat_dom)
    return results


def calc_initial_sate(ss_sys, ics, t0):
    print(">>> projecting initial states")
    u0 = [ics.pop(u) for u in ss_sys.inputs if u in ics]
    y0_orig = get_state(ics, ss_sys.orig_state, u0)
    if ss_sys.transformations:
        y0 = np.squeeze(ss_sys.transformations[0](t0, u0, y0_orig))
        return y0
    else:
        return y0_orig


def calc_original_state(sim_results, ss_sys, temp_dom):
    # check whether state has been altered
    if ss_sys.transformations is None:
        return sim_results

    # recover input trajectories
    input_traj = np.zeros((len(temp_dom), len(ss_sys.inputs)))
    for idx, inp in enumerate(ss_sys.inputs):
        vals = ss_sys.input_map[inp].get_results(temp_dom)
        input_traj[:, idx] = vals

    # recover transformed state trajectory
    state_traj = sim_results

    if ss_sys.transformations:
        # for now, use a loop
        orig_traj = np.zeros((len(temp_dom), len(ss_sys.orig_state)))
        for idx, t in enumerate(temp_dom):
            val = np.squeeze(ss_sys.transformations[1](t,
                                                       input_traj[idx],
                                                       state_traj[idx]))
            orig_traj[idx] = val
    else:
        orig_traj = state_traj

    return orig_traj, input_traj


def _convert_derivatives(new_forms):
    all_symbols = set()
    for f in new_forms:
        all_symbols.update(f.atoms(sp.Symbol))
    derivatives = _find_derivatives(new_forms, all_symbols)

    subs_list = []
    targets = set()
    for der in tqdm(derivatives):
        new_forms, subs_pairs, target_set = _convert_higher_derivative(
            new_forms, der)
        subs_list += subs_pairs
        targets.update(target_set)
    return new_forms, targets


def _find_weights(expressions):
    weights = set()
    for form in expressions:
        funcs = form.atoms(sp.Function)
        for f in funcs:
            if (isinstance(f.func, sp.function.UndefinedFunction)
                    and _weight_letter in str(f)
                    and len(f.args) == 1
                    and f.args[0] == time):
                weights.add(f)
    return weights


def _find_inputs(expressions):
    """ Find system inputs """
    inputs = set()
    for expr in expressions:
        for _inp in expr.atoms(sp.Function):
            if (isinstance(_inp.func, sp.function.UndefinedFunction)
                   and _input_letter in str(_inp.func)):
                inputs.add(_inp)

    return inputs


def _find_integrals(expressions):
    """ Find fake integrals """
    integrals = set()
    for eq in expressions:
        ints = [_int for _int in eq.atoms(FakeIntegral)]
        integrals.update(ints)

    return integrals


def _find_derivatives(weak_forms, sym):
    """ Find derivatives """
    if isinstance(sym, sp.Symbol):
        sym = {sym}

    derivatives = set()
    for eq in weak_forms:
        ders = {der for der in eq.atoms(sp.Derivative) if der.args[1] in sym}
        derivatives.update(ders)

    return derivatives


def _convert_higher_derivative(weak_forms, derivative):

    target_set = set()
    subs_list = []
    new_forms = weak_forms

    expr = derivative.args[0]
    order = len(derivative.args) - 1

    if _weight_letter in str(expr):
        if order > 1:
            # c1_dd = f()
            # c1_d = c2
            # -> c1_dd = c2_d
            # -> c2_d = f()
            new_var = get_weight()
            red_deriv = derivative.func(*derivative.args[:-1])
            new_eq = new_var - red_deriv
            new_der = new_var.diff(time)
            new_forms = weak_forms.xreplace({derivative: new_der})
            new_forms.append(new_eq)

            # replace remaining derivative
            new_forms, subs_pairs, new_targets = _convert_higher_derivative(
                new_forms, red_deriv)
            subs_list += subs_pairs
            target_set.update(new_targets)

            # replace newly introduced one
            new_forms, subs_pairs, new_targets = _convert_higher_derivative(
                new_forms, new_der)
            subs_list += subs_pairs
            target_set.update(new_targets)
        else:
            target_set.add(derivative)

    elif _input_letter in str(expr):
        warnings.warn("Temporal input derivative detected, this may cause"
                      "problems later on.")

    elif _function_letter in str(expr) or _test_function_letter in str(expr):
        if 1:
            callback = expr.func._imp_.derive(order)
            d_func = get_base_fraction_symbol(callback, *expr.args)
        else:
            # derive the associated callbacks
            if _function_letter in str(expr):
                d_func = get_function(*expr.args)

            elif _test_function_letter in str(expr):
                d_func = get_test_function(*expr.args)
            callback = expr.func._imp_
            d_func.func._imp_ = staticmethod(callback.derive(order))

        subs_pair = derivative, d_func
        new_forms = weak_forms.xreplace({derivative: d_func})
        subs_list += [subs_pair]
    else:
        raise NotImplementedError

    return new_forms, subs_list, target_set


def _simplify_integrals(weak_forms):

    integrals = _find_integrals(weak_forms)

    subs_dict = {}
    for integral in tqdm(integrals):
        red_int = _reduce_kernel(integral)
        if integral != red_int:
            subs_dict[integral] = red_int

    # red_forms = [f.subs(subs_list) for f in weak_forms]
    # red_forms = [f.xreplace(subs_dict) for f in weak_forms]
    red_forms = weak_forms.xreplace(subs_dict)
    return red_forms


def _solve_integrals(weak_forms):

    solved_map = dict()
    integrals = _find_integrals(weak_forms)
    for integral in tqdm(integrals):
        res = integral.eval_numerically()
        solved_map[integral] = res

    # tqdm.write(">>> substituting the solutions")
    subs_forms = weak_forms.xreplace(solved_map)

    return subs_forms


def _reduce_kernel(integral):
    """
    Move all independent terms outside the integral's kernel

    To reach this goal, firstly the kernel is expanded and split. Afterwards,
    independent term are moved aside. If the kernel still contains unknown
    expressions (such as weights and their derivatives) they are expanded into
    a taylor series and separated afterwards.
    The last step is only performed if `enable_approx` is `True` .

    Args:
        integral(FakeIntegral): Integral whose kernel is to be simplified.

    Returns:
        Sum of simplified integrals.

    """
    approx = get_parameter("enable_approx")

    kernel, (sym, a, b) = integral.args
    if time not in kernel.atoms(sp.Symbol):
        return integral

    kernel = kernel.expand()
    if isinstance(kernel, sp.Add):
        new_int = sp.Add(*[_reduce_kernel(FakeIntegral(addend, (sym, a, b)))
                           for addend in kernel.args])

    elif isinstance(kernel, sp.Mul):
        dep_args = []
        indep_args = []
        for arg in kernel.args:
            if sym not in arg.atoms(sp.Symbol):
                indep_args.append(arg)
            else:
                dep_args.append(arg)

        if not indep_args:
            bad_args = {arg for arg in dep_args if time in arg.free_symbols}
            if not bad_args:
                return FakeIntegral(kernel, (sym, a, b))
            else:
                if approx:
                    arg_map = {arg: _approximate_term(arg, sym)
                               for arg in bad_args}
                else:
                    return FakeIntegral(kernel, (sym, a, b))

            dep_args = [arg.xreplace(arg_map) for arg in dep_args]

        new_kernel = sp.Mul(*dep_args)
        new_part = _reduce_kernel(FakeIntegral(new_kernel, (sym, a, b)))
        new_int = sp.Mul(*indep_args) * new_part
    else:
        print("Unable to handle '{}'".format(kernel))
        return FakeIntegral(kernel, (sym, a, b))

    return new_int


def _approximate_term(term, sym):
    """
    Series expansion of inseparable terms

    Args:
        term: Term to expand.
        sym: Coordinate to expand.

    Returns:
        Taylor expansion of the highest order supported by the involved
        function expressions.
    """
    n = get_parameter("approx_order")
    pos = get_parameter("approx_pos")
    mode = get_parameter("approx_mode")

    warnings.warn("Approximating term {} at {}={} with order n={}".format(
        term, sym, pos, n))

    if mode == "series":
        res = sp.series(term, x=sym, x0=pos, n=n).removeO()
    elif mode == "pointwise":
        assert isinstance(term, sp.Function)
        weights = _find_weights([term])
        coeffs = {w: term.args[0].coeff(w) for w in weights}

        # rebuild expression
        res = sp.Add(*[coeffs[w] * term.func(w) for w in weights])

        residual = term.args[0] - sp.Add(*[coeffs[w] * w for w in weights])
        assert residual == 0

    return res


def _eliminate_input_derivatives(input_derivatives, weak_forms, orig_state, inputs):
    """
    Try to perform a generalized state transformation to eliminate derivatives
    Args:
        weak_forms:
        inputs:

    Returns:

    """
    new_state = sp.Matrix(get_weights(len(orig_state)))
    old_expr = new_state
    new_expr = orig_state
    neut_term = 0 * new_state
    for der in input_derivatives:
        dep, ret = _analyse_term_structure(weak_forms, der)
        if dep == "linear":
            new_expr += ret * der.args[0]
            old_expr -= ret * der.args[0]
            neut_term += ret * der
    # print(new_state)

    # substitute new state
    fwd_map = {old: new for old, new in zip(orig_state, new_expr)}
    rev_map = {new: old for new, old in zip(new_state, old_expr)}
    new_forms = weak_forms.xreplace(fwd_map) - neut_term
    # print(new_forms)

    # build generalized transformations
    inp_dummies = {inp: sp.Dummy() for inp in inputs}
    orig_dummies = {s: sp.Dummy() for s in orig_state}
    new_dummies = {s: sp.Dummy() for s in new_state}
    subs_map = {**inp_dummies, **orig_dummies, **new_dummies}

    fwd_args = ([time],
                [inp_dummies[s] for s in inputs],
                [orig_dummies[s] for s in orig_state])
    fwd_expr = new_expr.xreplace(subs_map)
    fwd_cb = sp.lambdify(fwd_args, expr=fwd_expr, modules="numpy")

    rev_args = ([time],
                [inp_dummies[s] for s in inputs],
                [new_dummies[s] for s in new_state])
    rev_expr = old_expr.xreplace(subs_map)
    rev_cb = sp.lambdify(rev_args, expr=rev_expr, modules="numpy")

    return new_forms, new_state, (fwd_cb, rev_cb)


def _analyse_term_structure(weak_forms, term):
    # try to get coefficient matrix
    coeffs = sp.Matrix([f.coeff(term) for f in weak_forms])
    if time not in coeffs.free_symbols:
        # bingo, linear dependency
        return "linear", coeffs
    elif not any([_weight_letter in str(s) for s in coeffs.free_symbols]):
        # some time dependent factors but now state members
        print(coeffs)
        raise NotImplementedError
    else:
        # multiplicative coupling, use inverse product rule
        print(coeffs)
        raise NotImplementedError


def simulate_state_space(ss_sys, y0, temp_dom):
    """
    Simulate an ODE system given its right-hand side

    """
    if 0:
        """
        JITCODE approch using compiled c code
        However, calling a controller implemented in python reduces the speed 
        gains
        """
        time_mapping = {time: t}

        def helper_factory(input_obj):
            def input_wrapper(_t, _y):
                return input_obj(time=t, weights=y)

            s_func = get_input(t, y)
            s_func.func._imp_ = input_wrapper

        input_mapping = {inp: helper_factory(ss_sys.input_map[inp])
                         for inp in ss_sys.inputs}
        state_mapping = {s: y(idx) for idx, s in enumerate(ss_sys.state)}
        subs_map = {**time_mapping, **input_mapping, **state_mapping}
        jitcode_rhs = ss_sys.rhs.xreplace(subs_map)
        ode = jitcode(jitcode_rhs)
        ode.set_integrator("lsoda")
        ode.set_initial_value(y0, temp_dom[0])

        print(">>> running time step simulation")
        np.seterr(under="warn")
        res = []
        t0 = clock()
        try:
            for _t in temp_dom:
                res.append(ode.integrate(_t))
        except UnsuccessfulIntegration:
            warnings.warn("Simulation failed at {}".format(_t))
        print("simulation took {}s".format(clock()-t0))

        return np.array(res)
    else:
        print(">>> building expressions")
        if 0:
            A, b = sp.linear_eq_to_matrix(ss_sys.rhs, *ss_sys.state)
            rhs = A @ ss_sys.state + b
        else:
            rhs = ss_sys.rhs

        input_mapping = {inp: sp.Dummy() for inp in ss_sys.inputs}
        state_mapping = {s: sp.Dummy() for s in ss_sys.state}
        subs_map = {**input_mapping, **state_mapping}

        dummy_rhs = rhs.xreplace(subs_map)

        args = [time,
                [input_mapping[inp] for inp in ss_sys.inputs],
                [state_mapping[st] for st in ss_sys.state]
                ]
        rhs_cb = sp.lambdify(args, expr=dummy_rhs, modules="numpy")

        def _rhs(_t, _q):
            # print(_t)
            _u = [ss_sys.input_map[inp](time=_t, weights=_q)
                  for inp in ss_sys.inputs]
            y_dt = np.ravel(rhs_cb(_t, _u, _q))
            return y_dt

        if 0:
            rhs_jac = dummy_rhs.jacobian(args[-1])
            jac_cb = sp.lambdify(args, expr=rhs_jac, modules="numpy")

            def _jac(t, y):
                u = [ss_sys.input_map[inp](time=t, weights=y)
                     for inp in ss_sys.inputs]
                jac = jac_cb(t, u, y)
                return jac

            return _rhs, _jac

        print(">>> running time step simulation")
        t_dom, res = old_ss_sim(_rhs, y0, temp_dom)
        return t_dom, res


def get_state(approx_map, state, extra_args):
    """
    Build initial state vector for time step simulation

    Args:
        approx_map(dict): Mapping that associates the used approximations with
            either symbolic expressions or lambda functions in the
            spatial dimensions.
        state(iterable): Iterable holding the elements of the state vector.

    Returns:
        Numpy array with shape (N,) where `N = len(state)` .

    """
    init_weights = dict()
    for key, val in approx_map.items():
        if isinstance(key, LumpedApproximation):
            _weight_set = key.approximate_function(val, *extra_args)
            new_d = {(lbl, w) for lbl, w in zip(key.weights, _weight_set)}
        elif _weight_letter in str(key):
            new_d = {key: val}
        else:
            raise NotImplementedError
        init_weights.update(new_d)
    y0 = np.array([init_weights[lbl] for lbl in state])
    return y0


def _sort_weights(weights, inp_values, ss_sys, approximations):
    """ Coordinate a given weight set with approximations """
    weight_dict = dict()
    extra_dict = dict()
    state_elements = list(ss_sys.state)
    inputs = list(ss_sys.inputs)
    for approx in approximations:
        a_lbls = approx.weights
        a_weights = []
        for lbl in a_lbls:
            idx = state_elements.index(lbl)
            a_weights.append(weights[:, idx])

        i_lbls = approx.extra_args
        i_values = []
        for lbl in i_lbls:
            idx = inputs.index(lbl)
            i_values.append(inp_values[:, idx])

        weight_dict[approx] = np.array(a_weights).T
        extra_dict[approx] = np.array(i_values).T

    return weight_dict, extra_dict


def _evaluate_approximations(weight_dict, extra_dict, approximations, temp_dom, spat_doms):
    """
    Evaluate approximations on the given grids
    """
    if isinstance(spat_doms, Domain):
        spat_doms = [spat_doms]

    results = []
    all_coords = [dom.points for dom in spat_doms]
    all_coords.append(np.array(range(len(temp_dom))))
    all_dims = [len(dom) for dom in all_coords]
    grids = np.meshgrid(*all_coords, indexing="ij")
    r_grids = [grid.ravel() for grid in grids]

    for approx in approximations:
        weight_mat = weight_dict[approx]
        extra_mat = extra_dict[approx]
        args = np.hstack((np.array(r_grids[:-1]).T,
                          weight_mat[r_grids[-1]]
                          ))
        if len(extra_mat) > 0:
            args = np.hstack((args, extra_mat[r_grids[-1]]))

        if 0:
            # lambdified functions do not like vectorial input
            res = approx(*args)
        else:
            res = np.zeros(len(r_grids[0]))
            for idx, row in enumerate(args):
                res[idx] = approx(*row)

        # per convention the time axis comes first
        out_data = np.moveaxis(res.reshape(all_dims), -1, 0)
        data = EvalData(input_data=[temp_dom] + spat_doms,
                        output_data=out_data)
        results.append(data)

    return results
