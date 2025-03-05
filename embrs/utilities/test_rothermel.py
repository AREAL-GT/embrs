import unittest
import numpy as np
from embrs.utilities.rothermel import (
    calc_propagation_in_cell,
    calc_r_0,
    get_characteristic_moistures,
    calc_live_mx,
    calc_I_r,
    calc_flux_ratio,
    calc_heat_sink,
    calc_r_and_i_along_dir,
    calc_E_B_C,
    calc_wind_factor,
    calc_slope_factor,
    calc_moisture_damping,
    calc_mineral_damping,
    calc_effective_wind_factor,
    calc_effective_wind_speed,
    calc_eccentricity
)
from embrs.utilities.fuel_models import Fuel, Anderson13, ScottBurgan40
from embrs.fire_simulator.cell import Cell

class TestRothermelModel(unittest.TestCase):
    def test_get_characteristic_moistures(self):
        # Fill in with appropriate inputs and expected outputs
        fuel = Anderson13(2)
        m_f = np.array([0.08, 0.10, 0.12, 0.35, 0.6])
        result = get_characteristic_moistures(fuel, m_f)
        expected = (0.0804, 0.95)  # Replace with expected output
        self.assertEqual(result, expected)

        fuel = Anderson13(13)

        print(f"w_n_lice: {fuel.w_n_live}")

        m_f = np.array([0.08, 0.10, 0.12, 0.35, 0.6])
        result = get_characteristic_moistures(fuel, m_f)
        expected = (0.086, 0.0)
        self.assertEqual(result, expected)


    def test_calc_live_mx(self):

        # Expected results from RMRS-GTR-371 p. 67

        fuel = ScottBurgan40(142) # SH2
        result = calc_live_mx(fuel, 0.06)
        expected = 0.98  # Replace with expected output
        self.assertAlmostEqual(result, expected, places=2)

        result = calc_live_mx(fuel, 0.10)
        expected = 0.44
        self.assertAlmostEqual(result, expected, places=2)

        result = calc_live_mx(fuel, 0.14)
        expected = 0.15
        self.assertAlmostEqual(result, expected, places=2)
        
        fuel = ScottBurgan40(146) # SH6
        result = calc_live_mx(fuel, 0.06)
        expected = 6.16  # Replace with expected output
        self.assertAlmostEqual(result, expected, places=2)

        result = calc_live_mx(fuel, 0.10)
        expected = 5.10
        self.assertAlmostEqual(result, expected, places=2)

        result = calc_live_mx(fuel, 0.14)
        expected = 4.03
        self.assertAlmostEqual(result, expected, places=2)

    def test_calc_wind_factor(self):
        # Fill in with appropriate inputs and expected outputs
        fuel = Anderson13(2)
        wind_speed_mph = 6
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        expected = 20.5  # Replace with expected output
        self.assertAlmostEqual(result, expected, delta=0.05)

        wind_speed_mph = 12
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        expected = 72.8  # Replace with expected output
        self.assertAlmostEqual(result, expected, delta=0.05)

        fuel = Anderson13(9)
        wind_speed_mph = 6
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        expected = 12.9  # Replace with expected output
        self.assertAlmostEqual(result, expected, delta=0.05)

        wind_speed_mph = 12
        wind_speed = wind_speed_mph * 88
        result = calc_wind_factor(fuel, wind_speed)
        expected = 42.7  # Replace with expected output
        self.assertAlmostEqual(result, expected, delta=0.05)


    def test_calc_slope_factor(self):
        # Fill in with appropriate inputs and expected outputs
        fuel = Anderson13(2)


        deg_11 = np.arctan(0.2)
        deg_27 = np.arctan(0.5)
        deg_45 = np.arctan(1)
        
        result = calc_slope_factor(fuel, deg_11)
        expected = 1.0 
        self.assertAlmostEqual(result, expected, delta=0.05)

        result = calc_slope_factor(fuel, deg_27)
        expected = 6.2
        self.assertAlmostEqual(result, expected, delta=0.05)

        result = calc_slope_factor(fuel, deg_45)
        expected = 24.9
        self.assertAlmostEqual(result, expected, delta=0.1)

        fuel = Anderson13(9)
        
        result = calc_slope_factor(fuel, deg_11)
        expected = 0.6 
        self.assertAlmostEqual(result, expected, delta=0.05)

        result = calc_slope_factor(fuel, deg_27)
        expected = 4.0
        self.assertAlmostEqual(result, expected, delta=0.05)

        result = calc_slope_factor(fuel, deg_45)
        expected = 16.0
        self.assertAlmostEqual(result, expected, delta=0.05)


    def test_calc_moisture_damping(self):
        # Dead tests
        m_f = 0.20
        m_x = 0.25
        result = calc_moisture_damping(m_f, m_x)
        expected = 0.4  # Replace with expected output
        self.assertAlmostEqual(result, expected, places=2)

        m_f = 0.20
        m_x = 0.45
        result = calc_moisture_damping(m_f, m_x)
        expected = 0.55  # Replace with expected output
        self.assertAlmostEqual(result, expected, places=2)
        
        m_f = 0.27
        m_x = 0.25
        result = calc_moisture_damping(m_f, m_x)
        expected = 0.0  # Replace with expected output
        self.assertAlmostEqual(result, expected, places=2)
        
        # Live tests
        fuel = ScottBurgan40(142)

        live_mx = calc_live_mx(fuel, 0.10)
        m_f = 0.30
        result = calc_moisture_damping(m_f, live_mx)
        expected = 0.49
        self.assertAlmostEqual(result, expected, places=2)

        m_f = 0.45
        result = calc_moisture_damping(m_f, live_mx)
        expected = 0.0
        self.assertAlmostEqual(result, expected, places=2)

        fuel = ScottBurgan40(146)
        live_mx = calc_live_mx(fuel, 0.10)
        m_f = 2.0
        result = calc_moisture_damping(m_f, live_mx)
        expected = 0.56
        self.assertAlmostEqual(result, expected, places=2)

    
    # def test_calc_propagation_in_cell(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     cell = Cell()
    #     R_h_in = None
    #     result = calc_propagation_in_cell(cell, R_h_in)
    #     expected = (np.array([]), np.array([]))  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_r_0(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     m_f = np.array([])
    #     result = calc_r_0(fuel, m_f)
    #     expected = (0.0, 0.0)  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_I_r(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     m_f = 0.0
    #     live_mf = 0.0
    #     result = calc_I_r(fuel, m_f, live_mf)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_flux_ratio(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     result = calc_flux_ratio(fuel)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_heat_sink(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     m_f = np.array([])
    #     result = calc_heat_sink(fuel, m_f)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_r_and_i_along_dir(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     cell = Cell()
    #     decomp_dir = 0.0
    #     R_h = 0.0
    #     I_r = 0.0
    #     alpha = 0.0
    #     e = 0.0
    #     result = calc_r_and_i_along_dir(cell, decomp_dir, R_h, I_r, alpha, e)
    #     expected = (0.0, 0.0)  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_E_B_C(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     result = calc_E_B_C(fuel)
    #     expected = (0.0, 0.0, 0.0)  # Replace with expected output
    #     self.assertEqual(result, expected)



    # def test_calc_mineral_damping(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     s_e = 0.010
    #     result = calc_mineral_damping(s_e)
    #     expected = 0.4174
    #     self.assertEqual(result, expected)

    # def test_calc_effective_wind_factor(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     R_h = 0.0
    #     R_0 = 0.0
    #     result = calc_effective_wind_factor(R_h, R_0)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_effective_wind_speed(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     R_h = 0.0
    #     R_0 = 0.0
    #     result = calc_effective_wind_speed(fuel, R_h, R_0)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

    # def test_calc_eccentricity(self):
    #     # Fill in with appropriate inputs and expected outputs
    #     fuel = Fuel()
    #     R_h = 0.0
    #     R_0 = 0.0
    #     result = calc_eccentricity(fuel, R_h, R_0)
    #     expected = 0.0  # Replace with expected output
    #     self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()