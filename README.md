# Laser Calibration for Locating Trapped Ions Suspended In An Electric Field (2D)

## Table of Contents
- [Description](#description)
- [Dependencies and Virtual Environment](#dependencies-and-virtual-environment)
- [Usage](#usage)
- [Testing](#testing)
- [Potential Improvements](#potential-improvements)

## Description

The contents of this directory are intended to locate the optimal mirror positions for eliciting a photon response from a suspended ion. This project locates the optimal mirror position for two mirrors using simulated ion response data, assuming a slightly noisy gaussian distribution.

#### Constraints:
```optimization_2d_multidirectional.py``` is designed with constraints on the range of possible mirror positions and ion response values. The mirror mount positions range from 0 to 1, where the true mirror positions might be normalized to this range in a practical setting. Similarly, the ion responses are constrained to the range of 0 to 100 photons per measurement round. 

#### Assumptions:
We leverage the assumption that the ion response would follow a gaussian distribution in order to fit the data with a gaussian function. By calculating the standard deviation of this curve in the x and y directions, we pinpoint a narrow range over which we focus our search for the curve's mean, which corresponds to the peak of the distribution and the optimal mirror positions. We then perform a grid search with an iteratively shortened step size until we locate the optimal mirror position with a specified precision. 

Additionally, we iteratively check the conditions that our current attempt at fitting generates a curve with an amplitude of between 80 and 100 photons per measurement round to utilize prior knowledge about the response distribution. With this, we set a range of acceptable amplitudes for the fitted curve to account for noise that might affect the response distribution's amplitude. The low-end of 80 photons per measurement round can be adjusted according to the amount of noise we might expect in a real-world scenario.

#### Strategy:
The script locates the peak ion response as follows:

- **STEP 1**: Attempt to generate a "valid fit" characterized as having an amplitude near 100. If unable, 1.25x the search range in the x and y directions and try again until successful, or both search ranges become [0, 1]. If the search range becomes [0, 1] in a direction, the corresponding step size will halve. Repeat STEP 1 until a valid fit is found. 

- **STEP 2**: From the fit, narrow the search range around 2 standard deviations for the x component, and 2 standard deviations for the y component. For each direction, set the new step size to the precision value.

- **STEP 3**: Try to find a new valid fit. If we cannot find a new valid fit that we can use to locate the mean (optimal mirror position) within a step size of a given precision or better, cut the step size in half and repeat STEP 3. 

- **STOP CONDITION**: Cancel the search entirely if the step size ever becomes lower than a specified minimum step size.

#### Notable Functionality:
All scans are multidirectional, meaning that the scan reverses course when it reaches an edge of the search space in a given direction.

##### Tunable Parameters:
```bash
x_start
x_stop
x_step_size
y_start
y_stop
y_step_size
precision # Desired minimum precision of the optimal mirror position
amp_min   # Minimum amplitude for a generated fit to be considered valid
minimum_step_size # Threshold step size that terminates program when reached
```

## Dependencies and Virtual Environment
To set up a virtual environment with the appropriate dependencies, run the following command in your project directory:
```bash
python setup.py
```

Activate your virtual environment depending on your operating system:
```bash
source laser_calibration/bin/activate # if using Unix
```

```bash
laser_calibration\Scripts\activate.bat # if using Windows
```

## Usage

After installing the dependencies, you can use the optimization script to locate the optimal mirror position. Fill in the stub functions for practical use.
```bash
python optimization_2d_multidirectional.py
```

## Testing

Use the unit test file entitled `unit_test_2d_multidirectional.py` to test the code in various scenarios:
```bash
python unit_test_2d_multidirectional.py
```

## Potential Improvements
It might be useful to perform a gradient descent method with an adaptive learning rate during STEP 3 instead of a grid search. However, this would present a challenge if the ion is positioned near the perimeter of the mirror search space. This would make it difficult for the adaptive learning rate to gain information by searching both sides of a distribution. 
