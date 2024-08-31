import unittest
from unittest.mock import patch
import numpy as np
from optimization_2d_multidirectional import locate_best_mirror_pos

class TestLocatePeak(unittest.TestCase):
    precision_check = 0.001

####################################################### Success checks. Return optimal mirror coordinates. #######################################################

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_center(self, mock_measure_ion_response): # (0.5, 0.5) position test (center position)
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 0.5, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (0.5 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 0.5, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (0.5 ± {self.precision_check})")
    
    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_zero_zero(self, mock_measure_ion_response): # (0,0) position test (corner position)
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0, y_mean=0, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 0, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (0 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 0, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (0 ± {self.precision_check})")

    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_zero_one(self, mock_measure_ion_response): # (1,0) position test (corner position)
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0, y_mean=1, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 0, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (0 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 1, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (1 ± {self.precision_check})")

    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_one_zero(self, mock_measure_ion_response): # (0,1) position test (corner position)
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=1, y_mean=0, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 1, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (1 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 0, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (0 ± {self.precision_check})")

    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_one_one(self, mock_measure_ion_response): # (1,1) position test (corner position)
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=1, y_mean=1, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 1, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (1 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 1, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (1 ± {self.precision_check})")

    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_one_one_and_small_initial_search(self, mock_measure_ion_response): # (1,1) position, far initial search window
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=1, y_mean=1, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=0.33, x_step_size=0.1, y_start=0, y_stop=0.33, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 1, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (1 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 1, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (1 ± {self.precision_check})")

    @patch('optimization_2d_multidirectional.measure_ion_response')
    def test_high_noise_level(self, mock_measure_ion_response): # high noise test
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=10)
        best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        self.assertAlmostEqual(best_x_pos, 0.5, delta=self.precision_check, msg=f"Optimal x mirror position {best_x_pos} out of expected range (0.5 ± {self.precision_check})")
        self.assertAlmostEqual(best_y_pos, 0.5, delta=self.precision_check, msg=f"Optimal y mirror position {best_y_pos} out of expected range (0.5 ± {self.precision_check})")

####################################################### Failure checks. Return None. #######################################################

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_x_start(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=-1, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_x_stop(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=2, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")


    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_x_step(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=-1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_y_start(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=-1, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_y_stop(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=2, x_step_size=0.1, y_start=0, y_stop=2, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_valid_y_step(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.11, y_start=0, y_stop=1, y_step_size=-1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

    @patch('optimization_2d_multidirectional.measure_ion_response') 
    def test_no_peak(self, mock_measure_ion_response): 
        mock_measure_ion_response.side_effect = lambda x_pos, y_pos: self.synthetic_response(x_pos, y_pos, x_mean=0.5, y_mean=0.5, amplitude=0, x_std_dev=0.1, y_std_dev=0.1, noise_level=1)
        result = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=self.precision_check, amp_min=80)
        if result is None:
            self.assertIsNone(result, "Expected None, but got something else.")
        else:
            best_x_pos, best_y_pos = result
            self.fail(f"Expected None, but got best_x_pos={best_x_pos}, best_y_pos={best_y_pos}")

##############################################################################################################

    def synthetic_response(self, x_pos, y_pos, x_mean=0, y_mean=0, amplitude=100, x_std_dev=0.1, y_std_dev=0.1, noise_level=1):
        """
        Generates synthetic ion response data based on gaussian function.
        This is a helper method to mock measure_ion_response.
        """
        gaussian_value = amplitude * np.exp(-(((x_pos - x_mean) ** 2) / (2 * x_std_dev ** 2) + ((y_pos - y_mean) ** 2) / (2 * y_std_dev ** 2)))
        noisy_response = gaussian_value + np.random.normal(0, noise_level)
        clipped_response = np.clip(noisy_response, 0, amplitude)        
        return int(round(clipped_response))

if __name__ == '__main__':
    unittest.main()
