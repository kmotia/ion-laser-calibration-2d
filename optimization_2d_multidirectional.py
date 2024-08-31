import numpy as np 
import matplotlib.pyplot as plt  
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit  

def gaussian_2d(pos, amplitude, x_mean, y_mean, x_std_dev, y_std_dev):
    """
    2D Gaussian function based on amplitude, x mean, y mean, x standard deviation, and y standard deviation

    Parameters:
    pos (tuple): Tuple with the x and y mirror positions.
    amplitude (float): The height of the Gaussian peak.
    x_mean (float): The x-coordinate of the peak.
    y_mean (float): The y-coordinate of the peak.
    x_std_dev (float): The standard deviation in the x-direction.
    y_std_dev (float): The standard deviation in the y-direction.

    Returns:
    np.ndarray: Gaussian values at each (x, y) position.
    """                                     
    x, y = pos
    return amplitude * np.exp(-(((x - x_mean) ** 2) / (2 * x_std_dev ** 2) + ((y - y_mean) ** 2) / (2 * y_std_dev ** 2)))

def measure_ion_response(x_pos, y_pos):
    """
    Gets the reading of an ion response for a given mirror position. (Just a stub for now).

    Parameters:
    x_pos (float): x Mirror position.
    y_pos (float): x Mirror position.
    """
    pass

def move_mirror_to_position(x_pos, y_pos):
    """
    Placeholder function to simulate moving the mirrors to specific positions. (Just a stub for now).
    
    Parameters:
    x_pos (float): Mirror position.
    y_pos (float): Mirror position.
    """
    pass

def locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=0.001, amp_min=80, min_step_size=0.00025):
    """
    Locates the peak of the ion response by iteratively measuring and fitting a 2D Gaussian.

    Parameters:
    x_start (float, optional): Starting x mirror position for the scan. Default is 0.
    x_stop (float, optional): Ending x mirror position for the scan. Default is 1.
    x_step_size (float, optional): Initial step size for scanning x mirror positions. Default is 0.1.
    y_start (float, optional): Starting Y mirror position for the scan. Default is 0.
    y_stop (float, optional): Ending Y mirror position for the scan. Default is 1.
    y_step_size (float, optional): Initial step size for scanning Y mirror positions. Default is 0.1.
    precision (float, optional): Desired precision for locating the optimal position of each mirror. Default is 0.001.
    amp_min (float, optional): Minimum acceptable amplitude for a valid fit. Default is 80.
    min_step_size (float, optional): Minimum step size to prevent overly fine searches. Default is 0.0005.

    Returns:
    tuple: Estimated x and Y mirror positions of the optimal mirror positions with required precision, or None if step size is too small.
    """
    # Check for valid search range
    if not (0 <= x_start <= 1) or not (0 <= x_stop <= 1) or not (0 <= y_start <= 1) or not (0 <= y_stop <= 1):
        print("Error: Start and stop values must be within the range [0, 1].")
        return None
    
    # Check if start is less than stop
    if (x_start >= x_stop) or (y_start >= y_stop):
        print("Error: Start values must be less than stop values")
        return None

    # Check for valid step size
    if x_step_size <= min_step_size or x_step_size >= (x_stop-x_start)/3 or y_step_size <= min_step_size or y_step_size >= (y_stop-y_start)/3:
        print("Error: Step size must be a positive value larger than 0, and no less than a 1/3 of the search range.")
        return None

    # Ensure start/stop values are within [0,1]
    x_start, x_stop = np.clip([x_start, x_stop], 0, 1)
    y_start, y_stop = np.clip([y_start, y_stop], 0, 1)

    def try_gaussian_fit(x_data, y_data, z_data):
        """
        Attempts to fit a 2D Gaussian to the ion response data.

        Parameters:
        x_data (numpy.ndarray): Array of x mirror positions.
        y_data (numpy.ndarray): Array of y mirror positions.
        z_data (numpy.ndarray): Array of ion response measurements.

        Returns:
        tuple or None: Optimal parameters of the Gaussian fit (amplitude, x_mean, y_mean, x_stddev, y_stddev) if successful, None otherwise.
        """

        if len(x_data) < 3 or len(y_data) < 3:
            return None
        try:
            popt, _ = curve_fit(gaussian_2d, (x_data, y_data), z_data,      # Uses non-linear least squares to fit a gaussian function to the data. It allows us to approximate data points that have not been physically measured. 
                                p0=[100, np.mean(x_data), np.mean(y_data), np.std(x_data), np.std(y_data)],
                                bounds = ([0, 0, 0, 0, 0],[100, 1, 1, np.inf, np.inf])) 
            return popt
        except RuntimeError:
            return None

    # Dictionary to store all of the data from searches
    ion_responses = {}
    # STEP 1: Initial search to find a valid fit using initial step size
    while True:

        x_direction = 1 # 1 means forward steps. -1 means backward steps
        y_direction = 1 # 1 means forward steps -1 means backward steps

        if x_direction == 1:
            x_vals = np.clip(np.arange(x_start, x_stop, x_step_size), 0, 1) 
        else:
            x_vals = np.clip(np.arange(x_start, x_stop, -x_step_size), 0, 1)

        for x_pos in x_vals:
            if x_pos not in ion_responses: # Only measure new x positions
                if y_direction == 1:
                    y_vals = np.clip(np.arange(y_start, y_stop, y_step_size), 0, 1)
                else:
                    y_vals = np.clip(np.arange(y_start, y_stop, -y_step_size), 0, 1)
                for y_pos in y_vals:
                    if y_pos not in ion_responses: # Only measure new y positions
                        # Gather ion response data over mirror positions
                        move_mirror_to_position(x_pos, y_pos)
                        ion_response = measure_ion_response(x_pos, y_pos)
                        ion_responses[(x_pos, y_pos)] = ion_response

            # End of y sweep. Reverse direction.
            y_direction *= -1
        # End of x sweep. Reverse direction.
        x_direction *= -1

        x_data = np.array([pos[0] for pos in ion_responses.keys()])
        y_data = np.array([pos[1] for pos in ion_responses.keys()])
        z_data = np.array(list(ion_responses.values()))

        # Attempt to fit a gaussian curve to the ion response data
        popt = try_gaussian_fit(x_data, y_data, z_data)

        # If we find a valid fit, break and move to STEP 2
        if popt is not None and amp_min < popt[0] <= 100:
            break

        # Expand the search range if we haven't found a valid fit with the current search parameters
        max_pos = max(ion_responses, key=ion_responses.get)
        x_range_width = x_stop - x_start
        y_range_width = y_stop - y_start
        x_start = np.clip(max_pos[0] - 1.25 * x_range_width, 0, 1)
        x_stop = np.clip(max_pos[0] + 1.25 * x_range_width, 0, 1)
        y_start = np.clip(max_pos[1] - 1.25 * y_range_width, 0, 1)
        y_stop = np.clip(max_pos[1] + 1.25 * y_range_width, 0, 1)

        # If search window has been expanded to entire space without finding a valid fit, reduce the step size
        if (x_start == 0 and x_stop == 1):
            x_step_size = x_step_size/2
        if (y_start == 0 and y_stop == 1):
            y_step_size = y_step_size/2
        if x_step_size <= min_step_size and y_step_size <= min_step_size:            
            print(f"Error: Step size became smaller than min_step_size, {min_step_size}.")
            return None
        
    # STEP 2: Narrow the search window around the peak of the gaussian fit
    amplitude, x_mean, y_mean, x_stddev, y_stddev = popt
    x_narrow_search_range = 2 * x_stddev
    y_narrow_search_range = 2 * y_stddev
    x_start = np.clip(x_mean - x_narrow_search_range, 0, 1)
    x_stop = np.clip(x_mean + x_narrow_search_range, 0, 1)
    y_start = np.clip(y_mean - y_narrow_search_range, 0, 1)
    y_stop = np.clip(y_mean + y_narrow_search_range, 0, 1)
    # Set step size for narrow search
    x_step_size = precision
    y_step_size = precision

    # STEP 3: Refine the search and validate fit and precision
    while True:

        x_direction = 1 # 1 means forward steps. -1 means backward steps
        y_direction = 1 # 1 means forward steps -1 means backward steps

        if x_direction == 1:
            x_vals = np.clip(np.arange(x_start, x_stop, x_step_size), 0, 1) 
        else:
            x_vals = np.clip(np.arange(x_start, x_stop, -x_step_size), 0, 1)

        for x_pos in x_vals:
            if x_pos not in ion_responses: # Only measure new x positions
                if y_direction == 1:
                    y_vals = np.clip(np.arange(y_start, y_stop, y_step_size), 0, 1)
                else:
                    y_vals = np.clip(np.arange(y_start, y_stop, -y_step_size), 0, 1)
                for y_pos in y_vals:
                    if y_pos not in ion_responses: # Only measure new y positions
                        # Gather ion response data over mirror positions
                        move_mirror_to_position(x_pos, y_pos)
                        ion_response = measure_ion_response(x_pos, y_pos)
                        ion_responses[(x_pos, y_pos)] = ion_response

            # End of y sweep. Reverse direction.
            y_direction *= -1
        # End of x sweep. Reverse direction.
        x_direction *= -1

        x_data = np.array([pos[0] for pos in ion_responses.keys()])
        y_data = np.array([pos[1] for pos in ion_responses.keys()])
        z_data = np.array(list(ion_responses.values()))

        # Attempt to fit a gaussian curve to the ion response data
        popt = try_gaussian_fit(x_data, y_data, z_data)

        # Check if we have a valid fit
        if popt is not None:
            amplitude, x_mean, y_mean, x_stddev, y_stddev = popt
            if amp_min < amplitude <= 100 and x_step_size < precision and y_step_size < precision:
                plot_results_2d(x_data, y_data, z_data, popt)
                return x_mean, y_mean

        # If valid fit condition or precision condition not met, halve the step size
        x_step_size /= 2
        y_step_size /= 2
        if x_step_size <= min_step_size and y_step_size <= min_step_size:
            print(f"Error: Step size became smaller than min_step_size, {min_step_size}.")
            return None
 
def plot_results_2d(x_data, y_data, z_data, popt):
    """
    Plots the measured ion response data and the fitted Gaussian curve.
    
    Parameters:
    x_data (list): x mirror position.
    y_data (list): y mirror position.
    z_data (list): Ion responses.
    popt (numpy.array): Optimal parameters of the Gaussian fit (amplitude, mean_x, mean_y, stddev_x, stddev_y).
    
    Returns:
    None
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_data, y_data, z_data, c='b', marker='o', alpha=0.3, label='Measured Data')
    
    # Generate a surface for the gaussian fit
    x_fit = np.linspace(0, 1, 1000)
    y_fit = np.linspace(0, 1, 1000)
    x_fit, y_fit = np.meshgrid(x_fit, y_fit)
    z_fit = gaussian_2d((x_fit, y_fit), *popt)
    ax.plot_surface(x_fit, y_fit, z_fit, cmap='Reds', alpha=0.6, rstride=100, cstride=100)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 120)
    
    # Manually set z-axis ticks and labels so that more of the green line locating optimal mirror positions shows in plot. This prevents z ticks past 100 from showing up.
    ax.set_zticks([i for i in range(0, 101, 20)]) 
    ax.set_zticklabels([str(i) for i in range(0, 101, 20)])
    
    # Proxy to make the gaussian surface labeled on the legend
    gaussian_proxy = Line2D([0], [0], linestyle="none", c='r', marker='o', alpha=0.4, label='Gaussian Surface Fit')
    
    # Line indicating optimal mirror positions
    optimal_line, = ax.plot([popt[1]] * 2, [popt[2]] * 2, [0, 120], color='green', linestyle='--')
    
    # Adding proxy to legend, and adding the green line after that
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([gaussian_proxy, optimal_line])  # Add the proxies to the legend
    labels.extend(['Gaussian Surface Fit', f'Optimal Mirr. Positions: x={popt[1]:0.4f}, y={popt[2]:0.4f}'])
    
    plt.title("Ion Response vs Mirror Position")
    ax.set_xlabel('x Mirror Position (unitless)')
    ax.set_ylabel('y Mirror Position (unitless)')
    ax.set_zlabel('Ion Response (photons/measurement round)')
    ax.legend(handles=handles, labels=labels, loc='upper left')
    plt.show()

# Example call to locate_best_mirror_pos
# best_x_pos, best_y_pos = locate_best_mirror_pos(x_start=0, x_stop=1, x_step_size=0.1, y_start=0, y_stop=1, y_step_size=0.1, precision=0.001, amp_min=80)
# print(f"Estimated optimal x and y mirror positions: x={best_x_pos}, y={best_y_pos}")
