import mbuild as mb

from mbuild.tests.base_test import BaseTest
from mbuild.exceptions import MBuildError
from copy import deepcopy
import warnings
import pytest
import numpy as np
from hypothesis import given
from hypothesis.strategies import integers, lists, floats
from hypothesis.extra.numpy import arrays


class TestHypo(BaseTest):
    def test_vwv_hypothesis(self, simple_cube):
        vec1 = arrays(np.float,3, elements=floats(-10000,10000)).example()
        vec2 = arrays(np.float, 3, elements=floats(-10000,10000)).example()
        print("v1, v2")
        print(vec1)
        print(vec2)
        test_lat = simple_cube
        test_lat.align_vector_with_vector(vec1, vec2)
        #beware nans
    def test_vwv_hypothesis_long(self, longx_cube):
        vec1 = arrays(np.float,3, elements=floats(-10000,10000)).example()
        vec2 = arrays(np.float, 3, elements=floats(-10000,10000)).example()
        print("v1, v2")
        print(vec1)
        print(vec2)
        test_lat = longx_cube
        test_lat.align_vector_with_vector(vec1, vec2)
        # beware nans

    def test_vwv_anchor_hypothesis(self, simple_cube):
        vec1 = arrays(np.float,3, elements=floats(-10000,10000)).example()
        vec2 = arrays(np.float, 3, elements=floats(-10000,10000)).example()
        anch = arrays(np.float, 3, elements=floats(-10000,10000)).example()
        print("v1, v2, anch")
        print(vec1)
        print(vec2)
        print(anch)
        test_lat = simple_cube
        test_lat.align_vector_with_vector(vec1, vec2, anchor_pt=anch)
        #beware nans

    def test_vwv_anchor_hypothesis_long(self, longx_cube):
        vec1 = arrays(np.float,3, elements=floats()).example()
        vec2 = arrays(np.float, 3, elements=floats(-100000,1000000)).example()
        anch = arrays(np.float, 3, elements=floats(-1000000,100000)).example()
        print("v1, v2, anch")
        print(vec1)
        print(vec2)
        print(anch)
        test_lat = longx_cube
        test_lat.align_vector_with_vector(vec1, vec2, anchor_pt=anch)
        # beware nans

    def test_vp_hypothesis_simple(self, simple_cube):
        for ii in range(5):
            vec1_1 = arrays(np.float,3, elements=floats(-100000,1000000)).example()
            vec1_2 = arrays(np.float, 3, elements=floats(-100000,100000)).example()
            vec2_1 = arrays(np.float,3, elements=floats(-10000,100000)).example()
            vec2_2 = arrays(np.float, 3, elements=floats(-100000,10000)).example()
            pair1 = [vec1_1, np.cross(vec1_1, vec1_2)]
            pair2 = [vec2_1, np.cross(vec2_1, vec2_2)]
            anch = arrays(np.float, 3, elements=floats(-10000,10000)).example()
            print("\np1,p2,anch")
            print(pair1)
            print(pair2)
            print(anch)
            simple_cube.align_vector_pairs(pair1, pair2, anchor_pt=anch)
        assert False
