import unittest
import pyinduct as pi
import numpy as np
import sympy as sp
import utils


class IntegrateFunctionClassTestClass(unittest.TestCase):
    """
    Test the integration of a function

    This class tests the initialisation of a IntegrateFunction and the method
    __call__(), which delivers the numerical integration value of the given
    function.
    """
    def setUp(self):
        self.assertRaises(TypeError, pi.Function, 42)
        self.sinfun = pi.Function(np.sin)
        self.t, self.x = sp.symbols('t x')

    def test_init(self):
        self.assertRaises(TypeError, utils.IntegrateFunction, 0, (0, 1))
        self.assertRaises(TypeError, utils.IntegrateFunction, self.sinfun,
                          (self.x, self.t))

    def test_call(self):
        intfunc = utils.IntegrateFunction(self.sinfun, (self.t, np.pi/2))
        self.assertAlmostEqual(intfunc(np.pi/4), 1/np.sqrt(2), places=7,
                               msg="Result of integration with variable limit"
                                   "is wrong!")
        intfunc = utils.IntegrateFunction(self.sinfun, (np.pi/2, self.t))
        self.assertAlmostEqual(intfunc(np.pi/4), -1/np.sqrt(2), places=7,
                               msg="Result of integration with variable limit"
                                   "is wrong!")
        intfunc = utils.IntegrateFunction(self.sinfun, (np.pi/4, np.pi/2))
        self.assertAlmostEqual(intfunc(), 1/np.sqrt(2), places=7,
                               msg="Result of integration with numerical limits"
                                   "is wrong!")
    def tearDown(self):
        pass
