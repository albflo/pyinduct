import unittest
import os
import numpy as np
import main
import main_newModel
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
        fileObject_sym = open("SSS_Sym_new", 'rb')
        self.A_sym_new = pickle.load(fileObject_sym)
        self.B_sym_new = pickle.load(fileObject_sym)
        fileObject_sym.close()
        fileObject_stef = open("SSS_Stef", 'rb')
        self.A_stef = pickle.load(fileObject_stef)
        self.B_stef = pickle.load(fileObject_stef)
        fileObject_stef.close()

    def test_compare_sym2stef(self):
        """
        compare symbolic framework to original one with model of stefan
        """
        np.testing.assert_almost_equal(self.A_sym, self.A_stef[1])
        np.testing.assert_almost_equal(self.B_sym, self.B_stef[0][1])

    def test_compare_new2stef(self):
        """
        compare new model to original one from stefan
        """
        N = int(self.A_sym.shape[0]/2)

        Asy = self.A_sym_new[N:2*N,:N]
        Asy_f = np.flip(np.flip(Asy, 0), 1)
        A_new_0 = np.hstack((self.A_sym_new[:N,:N], self.A_sym_new[:N,N:2*N]))
        A_new_1 = np.hstack((Asy_f, self.A_sym_new[N:2*N,N:2*N]))
        A_new = np.vstack((A_new_0, A_new_1))
        np.testing.assert_almost_equal(A_new, self.A_stef[1])

        Bsy = self.B_sym_new[N:2*N]
        Bsy_f = np.flip(Bsy, 0)
        B_new = np.array([np.hstack((self.B_sym_new[:N], Bsy_f))]).transpose()
        np.testing.assert_almost_equal(B_new, self.B_stef[0][1])

    def test_delete_files(self):
        # not really a test, only to delete the export files
        os.remove("SSS_Sym")
        os.remove("SSS_Sym_new")
        os.remove("SSS_Stef")

    def tearDown(self):
        pass
