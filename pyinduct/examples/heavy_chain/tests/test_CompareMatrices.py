import unittest
import os
import numpy as np
import main
import examples_stefan.hc_system_PI_officialVersion
import pickle

class CompareMatrices(unittest.TestCase):
    """
    Test the output fo the symbolic branch with the original simulation

    To test that the original model of Stefan is implemented with the symbolic
    framework and the original pyinduct framework. Both are exporting the
    matrices A and B of the state space system. These two files can be compared
    and checked.
    """

    def setUp(self):
        fileObject_sym = open("SSS_Sym", 'rb')
        self.A_sym = pickle.load(fileObject_sym)
        self.B_sym = pickle.load(fileObject_sym)
        fileObject_sym.close()
        os.remove("SSS_Sym")
        fileObject_stef = open("SSS_Stef", 'rb')
        self.A_stef = pickle.load(fileObject_stef)
        self.B_stef = pickle.load(fileObject_stef)
        fileObject_stef.close()
        os.remove("SSS_Stef")

    def test_compare(self):
        np.testing.assert_almost_equal(self.A_sym, self.A_stef[1])
        np.testing.assert_almost_equal(self.B_sym, self.B_stef[0][1])

    def tearDown(self):
        pass
