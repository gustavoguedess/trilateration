import json
import numpy as np
import math
import amr_params as amr
import functions.utils as utils

OUTLIER_MIN_DIST = 0.05
OUTLIER_MAX_DIST = 2

LIMIT_DIFF_OBJECT = 0.2 # Maximum difference between two samples to consider them as the same object
LIMIT_GAP = 3 # Number of samples between two objects to consider them as the same object

BEACON_RADIUS = 0.055

X = 0
Y = 1
THETA = 2

def print_objs(objs):
    for obj in objs:
        print(f"Object detected: {obj['middle_angle']:0.3f} ({utils.degrees(obj['middle_angle']):0.1f}º) {obj['distance']:0.2f}")
        # print(f"Start: {obj['start_angle']:0.2f} ({utils.degrees(obj['start_angle']):0.1f}º)")
        # print(f"End: {obj['end_angle']:0.2f} ({utils.degrees(obj['end_angle']):0.1f}º)")
        # print(f"Middle: {obj['middle_angle']:0.2f} ({utils.degrees(obj['middle_angle']):0.1f}º)")
        # print(f"Samples: {obj['samples']}")
        # print(f"Distance: {obj['distance']}")
        # print()

def calc_trilateration(data, last_state):
    # Remove outliers
    data = remove_outliers(data)
    if not any(data): raise Exception('There are no valid data')

    # Groupping objects
    objs_groups = group_data(data)

    # Get beacons
    beacons = get_beacons(objs_groups, last_state)

    log_register(objs_groups, beacons, data)

    if len(beacons) < 3: raise Exception('There are not enough beacons')

    # Get position and angle from beacons
    x,y = get_position(beacons)
    theta = calc_theta(beacons)

    return np.array([x, y, theta]).T

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
        start_angle_rad = utils.radians( _start/amr.LIDAR_RESOLUTION * 360 )
        end_angle_rad   = utils.radians(   _end/amr.LIDAR_RESOLUTION * 360 )
        sample_size = min(abs(_end-_start), samples-abs(_start-_end))
        
        obj = {
            'start_angle': start_angle_rad,
            'end_angle': end_angle_rad,
            'middle_angle': utils.circular_mean([start_angle_rad, end_angle_rad], normalize='2pi'),
            'samples': sample_size,
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

    def is_same_group(i, j):
        return is_same_size(i, j) and is_near(groups[i][1], groups[j][0])
    def merge_groups(i, j):
        groups[i] = (groups[i][0], groups[j][1])
        groups.pop(j)

    i = 0
    while i < len(groups):
        j = (i+1)%len(groups)
        if is_same_group(i, j):
            merge_groups(i, j)
        else:
            i += 1

    return groups

def get_distance_group(data, start, end):
    return np.median(data[start:end]) if start < end else np.median(data[start:] + data[:end])
    # return data[(start+end)//2]


##### GETTING BEACONS #####
def print_beacons(beacons):
    for key, beacon in beacons.items():
        if not beacon: print(f"{key}: None")
        else:
            print(f"{key}: {beacon['x']:0.2f}, {beacon['y']:0.2f} {beacon['phi']:0.3f}({utils.degrees(beacon['phi']):0.1f}º) {beacon['distance']:0.2f}")


def get_beacons(objs, last_state):
    global objects, beacons
    objects = objs.copy()
    last_beacons = calc_beacons(last_state)

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

def calc_beacons(state):
    beacons = {
        'beacon1': calc_beacon(state, amr.WHEELS_LEFT_X, amr.WHEELS_UPPER_Y),
        'beacon2': calc_beacon(state, amr.WHEELS_LEFT_X, amr.WHEELS_BOTTOM_Y),
        'beacon3': calc_beacon(state, amr.WHEELS_RIGHT_X, amr.WHEELS_BOTTOM_Y),
        'beacon4': calc_beacon(state, amr.WHEELS_RIGHT_X, amr.WHEELS_UPPER_Y),
    }
    return beacons

def calc_beacon(state, bx, by):
    return {
        'x': bx,
        'y': by,
        'phi': calc_phi(state, bx, by),
        'distance': euclidean_distance(state[X], state[Y], bx, by),
    }

# Calculate the phi angle in radians between the trilateration and the beacon. The angle is between 0 and 2pi
def calc_phi(trilateration, bx, by):
    dy = by - trilateration[Y]
    dx = bx - trilateration[X]
    theta = trilateration[THETA]
    
    gamma = np.arctan2(dy, dx)
    # theta = trilateration['theta'] if trilateration['theta'] >= 0 else 360 + trilateration['theta']
    # theta = math.radians(theta)

    phi = utils.angle_diff(gamma, theta)
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return phi

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

        limit_error_distance = error_distance_beacon(possible_beacon['distance'])
        limit_error_phi = error_phi_beacon(possible_beacon['phi']) 

        diff_distance = abs(possible_beacon['distance'] - beacon['distance'])
        diff_phi = utils.angle_diff(possible_beacon['phi'], beacon['phi'])
        
        if np.abs(diff_distance) < limit_error_distance and np.abs(diff_phi) < limit_error_phi:
            possible_beacons_list.append(possible_beacon)

    if len(possible_beacons_list) == 1:
        return possible_beacons_list[0]
    
    return None


#TODO: Improve this function
def error_distance_beacon(distance):
    return distance * 0.15

#TODO: Improve this function
def error_phi_beacon(phi):
    return np.radians(15) # 20 degrees in radians of error

def obj_to_beacon(obj, last_beacon):
    phi = obj['middle_angle']
    distance = calc_beacon_distance(obj['data'])
    
    #TODO check phi and distance are compatible
    # cos_phi = BEACON_RADIUS/distance
    # if abs(np.cos(phi_rad) - cos_phi) > 0.1: return None

    return {
        'x': last_beacon['x'],
        'y': last_beacon['y'],
        'phi': phi,
        'distance': distance,
    }

#TODO: Improve this function
def calc_beacon_distance(distances):
    distance = np.median(distances)
    return distance+BEACON_RADIUS


##### GETTING POSITION #####

# Return the y position of the beacon. The both beacons must have the same x position
def calc_y(b1,b2):
    if 'distance' not in b1 or 'distance' not in b2: return None
    return (b1['distance']**2 - b2['distance']**2 + b2['y']**2 - b1['y']**2) / (2*b2['y'] - 2*b1['y'])

# Return the x position of the beacon. The both beacons must have the same y position
def calc_x(b1,b2):
    if 'distance' not in b1 or 'distance' not in b2: return None
    return (b1['distance']**2 - b2['distance']**2 + b2['x']**2 - b1['x']**2) / (2*b2['x'] - 2*b1['x'])

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
    phi_1 = beacons['beacon1']['phi'] if 'phi' in beacons['beacon1'] else 0
    phi_2 = beacons['beacon2']['phi'] if 'phi' in beacons['beacon2'] else 0
    phi_3 = beacons['beacon3']['phi'] if 'phi' in beacons['beacon3'] else 0
    phi_4 = beacons['beacon4']['phi'] if 'phi' in beacons['beacon4'] else 0

    # diff phi between beacons
    delta_12 = utils.angle_diff(phi_2, phi_1)
    delta_23 = utils.angle_diff(phi_3, phi_2)
    delta_34 = utils.angle_diff(phi_4, phi_3)
    delta_41 = utils.angle_diff(phi_1, phi_4)

    # gamma
    b1_gamma   =  (np.pi/2   + utils.arcsin( np.sin(delta_12) * dist_2 / dist_12 ) ) % (2*np.pi)
    b2_gamma   =  (np.pi     + utils.arcsin( np.sin(delta_23) * dist_3 / dist_23 ) ) % (2*np.pi)
    b3_gamma   =  (3*np.pi/2 + utils.arcsin( np.sin(delta_34) * dist_4 / dist_34 ) ) % (2*np.pi)
    b4_gamma   =  (            utils.arcsin( np.sin(delta_41) * dist_1 / dist_41 ) ) % (2*np.pi)
    b1_gamma_l =  (np.pi     - utils.arcsin( np.sin(delta_41) * dist_4 / dist_41 ) + 2*np.pi) % (2*np.pi)
    b2_gamma_l =  (3*np.pi/2 - utils.arcsin( np.sin(delta_12) * dist_1 / dist_12 ) + 2*np.pi) % (2*np.pi)
    b3_gamma_l =  (2*np.pi   - utils.arcsin( np.sin(delta_23) * dist_2 / dist_23 ) + 2*np.pi) % (2*np.pi)
    b4_gamma_l =  (np.pi/2   - utils.arcsin( np.sin(delta_34) * dist_3 / dist_34 ) + 2*np.pi) % (2*np.pi)

    # theta
    b1_theta    = utils.angle_diff(b1_gamma   , phi_1)
    b2_theta    = utils.angle_diff(b2_gamma   , phi_2)
    b3_theta    = utils.angle_diff(b3_gamma   , phi_3)
    b4_theta    = utils.angle_diff(b4_gamma   , phi_4)
    b1_theta_l  = utils.angle_diff(b1_gamma_l , phi_1)
    b2_theta_l  = utils.angle_diff(b2_gamma_l , phi_2)
    b3_theta_l  = utils.angle_diff(b3_gamma_l , phi_3)
    b4_theta_l  = utils.angle_diff(b4_gamma_l , phi_4)
    
    theta_list = [b1_theta, b2_theta, b3_theta, b4_theta, b1_theta_l, b2_theta_l, b3_theta_l, b4_theta_l]

    theta = utils.circular_mean(theta_list)

    # Normalize to [-np.pi, np.pi]
    theta = utils.normalize_to_pi_radian(theta)

    if np.isnan(theta): 
        breakpoint()

    return theta


#TODO DEBUG

def init_log(filename='trilateration.log'):
    # delete the file if it exists
    with open(filename, 'w') as f:
        f.write('')  # Clear the file

def log_register(objs, beacons, values, filename='trilateration.log'):
    log_data = {
        'objs': objs,
        'beacons': beacons,
        'values': values
    }

    with open(filename, 'a') as f:
        f.write(json.dumps(log_data) + '\n')


# Generate a image with all objects detected
def generate_image(objs, beacons):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    for obj in objs:
        start_angle = obj['start_angle']
        end_angle = obj['end_angle']
        middle_angle = obj['middle_angle']
        distance = obj['distance']

        # Draw the object as a line
        x_start = distance * np.cos(start_angle)
        y_start = distance * np.sin(start_angle)
        x_end = distance * np.cos(end_angle)
        y_end = distance * np.sin(end_angle)

        ax.plot([x_start, x_end], [y_start, y_end], 'r-')

    for key, beacon in beacons.items():
        if not beacon: continue
        phi = beacon['phi']
        distance = beacon['distance']

        # Calculate the x and y coordinates of the beacon relative to the center (ignore x and y of the beacon)
        x = distance * np.cos(phi)
        y = distance * np.sin(phi)

        # Draw the beacon as a point tinyer than the object
        ax.plot(x, y, 'bo', markersize=2)

        # Draw error circle
        distance = error_distance_beacon(distance)
        circle = plt.Circle((x, y), distance, color='g', fill=False)
        ax.add_patch(circle)


    # Plot current position with a red dot
    ax.plot(0, 0, 'ro')
    ax.set_aspect('equal', adjustable='box')

    # plt.show()
    plt.savefig('objs.png')
    plt.close(fig)
