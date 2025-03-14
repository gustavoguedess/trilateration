import numpy as np 

def angle_normalize(theta):
    return (theta + 2*np.pi) % (4 * np.pi) - np.pi

def min_angle_diff(a, b):
    a = angle_normalize(a)
    b = angle_normalize(b)
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff if diff != -np.pi else np.pi


def degree2radian(degrees):
    return np.radians((degrees + 360) % 360)

def radian2degree(radians):
    degrees = np.degrees(radians)
    return (degrees + 180) % 360 - 180

def normalize_radian(radians):
    return np.mod(radians, 2*np.pi)



if __name__ == '__main__':
    example_convert_degrees = [
        0,
        90,
        180,
        -180,
        -90
    ]

    example_convert_radians = [
        0,
        np.pi/2,
        np.pi,
        3*np.pi/2,
        2*np.pi,
    ]

    example_normalize_radians = [
        0,
        np.pi/2,
        np.pi,
        3*np.pi/2,
        2*np.pi,
        2*np.pi + np.pi/2,
        2*np.pi + np.pi,
        -np.pi/2,
        -np.pi,
        -3*np.pi/2,
    ]
    
    print(f'Convert degrees to radians: ')
    for degree in example_convert_degrees:
        print(f'{degree}° = {degrees2radians(degree)} rad')

    print(f'\nConvert radians to degrees: ', len(example_convert_radians))
    for radian in example_convert_radians:
        print(f'{radian} rad = {radians2degrees(radian)}°')

    print(f'\nNormalize radians: ')
    for radian in example_normalize_radians:
        print(f'{radian} rad = {normalize_radians(radian)} rad')