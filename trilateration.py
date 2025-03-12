import json
import numpy as np
import math
import amr_params as amr

OUTLIER_MIN_DIST = 0.05
OUTLIER_MAX_DIST = 2

LIMIT_DIFF_OBJECT = 0.2 # Maximum difference between two samples to consider them as the same object
LIMIT_GAP = 3 # Number of samples between two objects to consider them as the same object

BEACON_RADIUS = 0.055


def calc_trilateration(data, last_trilateration):
    # Remove outliers
    data = remove_outliers(data)
    if not any(data): raise Exception('There are no valid data')

    # Groupping objects
    objs_groups = group_data(data)

    # Get beacons
    beacons = get_beacons(objs_groups, last_trilateration)

    if len(beacons) < 3: raise Exception('There are not enough beacons')

    # Get position and angle from beacons
    x,y = get_position(beacons)
    theta = calc_theta(beacons)

    trilateration = {
        'x': x,
        'y': y,
        'theta': theta,
    }

    return trilateration


##### REMOVING OUTLIERS #####
# Remove outliers from the data
def remove_outliers(data):
    data = data.copy()

    for i in range(len(data)):
        if   data[i] < OUTLIER_MIN_DIST: data[i] = 0
        elif data[i] > OUTLIER_MAX_DIST: data[i] = OUTLIER_MAX_DIST
    
    return data

##### GROUPING DATA #####
# Return a list of objects range detected by the lidar sensor
def group_data(data):
    samples = len(data)
    ranges = dumb_group(data)
    ranges = union_similar_objs(data, ranges)
    #TODO I think I can merge these two functions

    groups = []
    for _start, _end in ranges:
        # _start to angle
        _start = (_start * 360) // samples
        _end = (_end * 360) // samples
        obj = {
            'start': _start,
            'end': _end,
            'middle': ((_start+_end)//2)%360,
            'size': abs(_end-_start),
            'distance': get_distance_group(data, _start, _end) + 0.055, #TODO magic number is equivalent to the beacon radius
            'data': data[_start:_end] if _start < _end else data[_start:] + data[:_end]
        }
        groups.append(obj)
    return groups

### GROUPING OBJECTS ###
def dumb_group(data):
    groups = []
    i = 0
    while i < len(data):
        if data[i] != 0:
            start_object = i
            end_object = i+1
            while end_object < len(data) and data[end_object] != 0 and abs(data[end_object]-data[end_object-1]) < LIMIT_DIFF_OBJECT:
                end_object += 1
            groups.append((start_object, end_object))
            i = end_object
        else:
            i += 1
    return groups


def union_similar_objs(data, groups):
    groups = groups.copy()

    is_same_size = lambda i, j: abs(data[groups[i][1]-1]-data[groups[j][0]]) < LIMIT_DIFF_OBJECT
    is_near = lambda i, j: j-i < LIMIT_GAP if i < j else len(data)-i+j < LIMIT_GAP
    i = 0
    while i < len(groups):
        if is_same_size(i, (i+1)%len(groups)) and is_near(groups[i][1], groups[(i+1)%len(groups)][0]):
            groups[i] = (groups[i][0], groups[(i+1)%len(groups)][1])
            groups.pop((i+1)%len(groups))
        else:
            i += 1
    return groups

def get_distance_group(data, start, end):
    return data[(start+end)//2]


##### GETTING BEACONS #####
def get_beacons(objs, last_trilateration):
    last_beacons = calc_beacons(last_trilateration)

    beacons = {
        'beacon1': get_similar_beacon(objs, last_beacons['beacon1']),
        'beacon2': get_similar_beacon(objs, last_beacons['beacon2']),
        'beacon3': get_similar_beacon(objs, last_beacons['beacon3']),
        'beacon4': get_similar_beacon(objs, last_beacons['beacon4']),
    }

    if not beacons['beacon1']: beacons['beacon1'] = last_beacons['beacon1']
    if not beacons['beacon2']: beacons['beacon2'] = last_beacons['beacon2']
    if not beacons['beacon3']: beacons['beacon3'] = last_beacons['beacon3']
    if not beacons['beacon4']: beacons['beacon4'] = last_beacons['beacon4']

    return beacons

def calc_beacons(trilateration):
    beacons = {
        'beacon1': calc_beacon(trilateration, amr.WHEELS_LEFT_X, amr.WHEELS_UPPER_Y),
        'beacon2': calc_beacon(trilateration, amr.WHEELS_LEFT_X, amr.WHEELS_BOTTOM_Y),
        'beacon3': calc_beacon(trilateration, amr.WHEELS_RIGHT_X, amr.WHEELS_BOTTOM_Y),
        'beacon4': calc_beacon(trilateration, amr.WHEELS_RIGHT_X, amr.WHEELS_UPPER_Y),
    }
    return beacons

def calc_beacon(trilateration, bx, by):
    return {
        'x': bx,
        'y': by,
        'phi': np.degrees(calc_phi(trilateration, bx, by)),
        'distance': euclidean_distance(trilateration['x'], trilateration['y'], bx, by),
    }

# Calculate the phi angle in radians between the trilateration and the beacon. The angle is between 0 and 2pi
def calc_phi(trilateration, bx, by):
    dy = by - trilateration['y']
    dx = bx - trilateration['x']
    gamma = math.atan2(dy, dx)
    theta = trilateration['theta'] if trilateration['theta'] >= 0 else 360 + trilateration['theta']
    theta = math.radians(theta)
    phi = diff_angle(gamma, theta)
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return phi


# Return the difference between two angles in radians (-pi, pi)
def diff_angle(angle1, angle2):
    diff = (angle1 - angle2 + 2 * np.pi) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff


def euclidean_distance(*args):
    if len(args) == 2:
        a_x, a_y, b_x, b_y = args[0]['x'], args[0]['y'], args[1]['x'], args[1]['y']
    elif len(args) == 4:
        a_x, a_y, b_x, b_y = args
    else: return None

    return math.sqrt((a_x-b_x)**2 + (a_y-b_y)**2)


def get_similar_beacon(objs, beacon):
    possible_beacons_list = []

    for obj in objs:
        possible_beacon = obj_to_beacon(obj, beacon)
        if not possible_beacon: continue

        error_distance = error_distance_beacon(possible_beacon['distance'])
        error_phi = error_phi_beacon(possible_beacon['phi']) 

        diff_distance = abs(possible_beacon['distance'] - beacon['distance'])
        diff_phi = np.degrees(diff_angle(np.radians(possible_beacon['phi']), np.radians(beacon['phi'])))

        if np.abs(diff_distance) < error_distance \
            and np.abs(diff_phi) < error_phi:
            possible_beacons_list.append(possible_beacon)    

    if len(possible_beacons_list) == 1:
        return possible_beacons_list[0]
    
    return None


#TODO: Improve this function
def error_distance_beacon(distance):
    return distance * 0.1

#TODO: Improve this function
def error_phi_beacon(phi):
    return 15

def obj_to_beacon(obj, last_beacon):
    phi = np.degrees(circular_mean([np.radians(obj['start']), np.radians(obj['end'])]))
    phi_rad = np.radians(phi)
    distance = calc_beacon_distance(obj['data'])
    
    # check phi and distance are compatible
    cos_phi = BEACON_RADIUS/distance
    # if abs(np.cos(phi_rad) - cos_phi) > 0.1: return None

    return {
        'x': last_beacon['x'],
        'y': last_beacon['y'],
        'phi': phi,
        'distance': distance,
    }


def circular_mean(angles):
    x = np.mean(np.cos(angles))
    y = np.mean(np.sin(angles))
    theta = np.arctan2(y, x)
    theta_normalized = (theta + 2 * np.pi) % (2 * np.pi)
    return theta_normalized

#TODO: Improve this function
def calc_beacon_distance(distances):
    middle = len(distances) // 2
    return distances[middle]+BEACON_RADIUS


##### GETTING POSITION #####

# Return the y position of the beacon. The both beacons must have the same x position
def calc_y(b1,b2):
    if 'distance' not in b1 or 'distance' not in b2: return None
    return (b1['distance']**2 - b2['distance']**2 + b2['y']**2 - b1['y']**2) / (2*b2['y'] - 2*b1['y'])

# Return the x position of the beacon. The both beacons must have the same y position
def calc_x(b1,b2):
    if 'distance' not in b1 or 'distance' not in b2: return None
    return (b1['distance']**2 - b2['distance']**2 + b2['x']**2 - b1['x']**2) / (2*b2['x'] - 2*b1['x'])

#TODO Verify later, this function is not working properly
def asin(x):
    if x < -1: x = -1
    if x > 1: x = 1
    result = np.arcsin( x )
    result = (result + 2 * np.pi) % (2 * np.pi)
    return result

def get_position(beacons):    
    x1 = calc_x(beacons['beacon2'], beacons['beacon3']) if 'beacon2' in beacons and 'beacon3' in beacons else None
    x2 = calc_x(beacons['beacon1'], beacons['beacon4']) if 'beacon1' in beacons and 'beacon4' in beacons else None
    y1 = calc_y(beacons['beacon2'], beacons['beacon1']) if 'beacon2' in beacons and 'beacon1' in beacons else None
    y2 = calc_y(beacons['beacon3'], beacons['beacon4']) if 'beacon3' in beacons and 'beacon4' in beacons else None

    x = [x1, x2] 
    y = [y1, y2]

    # remove if there is none value
    x = [i for i in x if i]
    y = [i for i in y if i]
    
    if len(x) == 0 or len(y) == 0: return None, None

    x = sum(x) / len(x)
    y = sum(y) / len(y)

    return x, y

def calc_theta(beacons):
    # distances
    dist_12 = euclidean_distance(beacons['beacon1'], beacons['beacon2'])
    dist_23 = euclidean_distance(beacons['beacon2'], beacons['beacon3'])
    dist_34 = euclidean_distance(beacons['beacon3'], beacons['beacon4'])
    dist_41 = euclidean_distance(beacons['beacon4'], beacons['beacon1'])

    dist_1 = beacons['beacon1']['distance'] if 'distance' in beacons['beacon1'] else 0
    dist_2 = beacons['beacon2']['distance'] if 'distance' in beacons['beacon2'] else 0
    dist_3 = beacons['beacon3']['distance'] if 'distance' in beacons['beacon3'] else 0
    dist_4 = beacons['beacon4']['distance'] if 'distance' in beacons['beacon4'] else 0

    # phi
    phi_1 = np.radians(beacons['beacon1']['phi']) if 'phi' in beacons['beacon1'] else 0
    phi_2 = np.radians(beacons['beacon2']['phi']) if 'phi' in beacons['beacon2'] else 0
    phi_3 = np.radians(beacons['beacon3']['phi']) if 'phi' in beacons['beacon3'] else 0
    phi_4 = np.radians(beacons['beacon4']['phi']) if 'phi' in beacons['beacon4'] else 0

    # diff phi between beacons
    delta_12 = diff_angle(phi_2, phi_1)
    delta_23 = diff_angle(phi_3, phi_2)
    delta_34 = diff_angle(phi_4, phi_3)
    delta_41 = diff_angle(phi_1, phi_4)
    
    # gamma
    b1_gamma   =  (np.pi/2   + asin( np.sin(delta_12) * dist_2 / dist_12 ) ) % (2*np.pi)
    b2_gamma   =  (np.pi     + asin( np.sin(delta_23) * dist_3 / dist_23 ) ) % (2*np.pi)
    b3_gamma   =  (3*np.pi/2 + asin( np.sin(delta_34) * dist_4 / dist_34 ) ) % (2*np.pi)
    b4_gamma   =  (            asin( np.sin(delta_41) * dist_1 / dist_41 ) ) % (2*np.pi)
    b1_gamma_l =  (np.pi     - asin( np.sin(delta_41) * dist_4 / dist_41 ) + 2*np.pi) % (2*np.pi)
    b2_gamma_l =  (3*np.pi/2 - asin( np.sin(delta_12) * dist_1 / dist_12 ) + 2*np.pi) % (2*np.pi)
    b3_gamma_l =  (2*np.pi   - asin( np.sin(delta_23) * dist_2 / dist_23 ) + 2*np.pi) % (2*np.pi)
    b4_gamma_l =  (np.pi/2   - asin( np.sin(delta_34) * dist_3 / dist_34 ) + 2*np.pi) % (2*np.pi)

    # theta
    b1_theta    = diff_angle(b1_gamma   , phi_1)
    b2_theta    = diff_angle(b2_gamma   , phi_2)
    b3_theta    = diff_angle(b3_gamma   , phi_3)
    b4_theta    = diff_angle(b4_gamma   , phi_4)
    b1_theta_l  = diff_angle(b1_gamma_l , phi_1)
    b2_theta_l  = diff_angle(b2_gamma_l , phi_2)
    b3_theta_l  = diff_angle(b3_gamma_l , phi_3)
    b4_theta_l  = diff_angle(b4_gamma_l , phi_4)
    
    theta_list = [b1_theta, b2_theta, b3_theta, b4_theta, b1_theta_l, b2_theta_l, b3_theta_l, b4_theta_l]

    theta_rad = circular_mean(theta_list)

    theta_degree = np.degrees(theta_rad)

    theta = theta_degree if theta_degree < 180 else theta_degree - 360

    if theta is None: 
        return None
    return theta