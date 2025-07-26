import numpy as np 


def normalize_to_pi_radian(angle_radians):
    """ Normalize the radian value to be between -pi and pi """
    return (angle_radians + np.pi) % (2 * np.pi) - np.pi

def normalize_to_2pi_radian(angle_radians):
    """ Normalize the radian value to be between 0 and 2pi """
    return angle_radians % (2 * np.pi)

def normalize_to_360_degree(angle_degrees):
    """ Normalize the degree value to be between 0 and 360 """
    return angle_degrees % 360

def normalize_to_180_degree(angle_degrees):
    """ Normalize the degree value to be between -180 and 180 """
    return (angle_degrees + 180) % 360 - 180

def radians(angle_degrees, normalize=False):
    """ Convert degrees to radians """
    if not normalize:
        return np.radians(angle_degrees)
    elif normalize == '2pi':
        return normalize_to_2pi_radian(np.radians(angle_degrees))
    elif normalize == 'pi':
        return normalize_to_pi_radian(np.radians(angle_degrees))
    else:
        raise ValueError(f"Invalid normalize value: {normalize}")
    
def degrees(angle_radians, normalize=False):
    """ Convert radians to degrees """
    if not normalize:
        return np.degrees(angle_radians)
    elif normalize == '2pi':
        return normalize_to_2pi_radian(np.degrees(angle_radians))
    elif normalize == 'pi':
        return normalize_to_pi_radian(np.degrees(angle_radians))
    else:
        raise ValueError(f"Invalid normalize value: {normalize}")

# def degree_to_radian_normalized(degrees):
#     return normalize_radian(np.radians(degrees))

# def radian_to_degree(radians):
#     return np.degrees(normalize_radian(radians))


def angle_diff(a, b):
    """ Calculate the difference between two angles """
    a = normalize_to_pi_radian(a)
    b = normalize_to_pi_radian(b)
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi

    return normalize_to_pi_radian(diff)



# def circular_mean(angles):
#     x = np.mean(np.cos(angles))
#     y = np.mean(np.sin(angles))
#     theta = np.arctan2(y, x)
#     theta_normalized = (theta + 2 * np.pi) % (2 * np.pi)
#     return theta_normalized


#TODO Verify later, this function is not working properly
def arcsin(x):
    if x < -1: x = -1
    if x > 1: x = 1
    result = np.arcsin( x )
    result = normalize_to_2pi_radian(result)
    return result

def circular_mean(angles, normalize=False):
    """ Calculate the mean angle of a list of angles """
    # Convert angles to x, y coordinates
    x = np.cos(angles)
    y = np.sin(angles)

    # Calculate the mean of the x and y coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Convert the mean x, y coordinates back to an angle
    theta = np.arctan2(y_mean, x_mean)

    if not normalize:
        return theta
    elif normalize == '2pi':
        return normalize_to_2pi_radian(theta)
    elif normalize == 'pi':
        return normalize_to_pi_radian(theta)
    else:
        raise ValueError(f"Invalid normalize value: {normalize}")
