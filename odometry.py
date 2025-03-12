import numpy as np 
import amr_params as amr

def calc_amr_velocity(time, enc1, enc2, enc3, enc4, r=0.1, l=0.3, w=0.1):
    l = amr.WHEELS_LENGTH
    w = amr.WHEELS_WIDTH
    
    enc = np.array([enc1, enc2, enc3, enc4]).T

    u = enc*2*np.pi/(30*64*2*time) # 30 is the gear ratio, 64 is the encoder resolution. 2 is a magical number that makes it work
    
    H = 1/r * np.array([
        [-l+w, 1, -1],
        [l+w, 1, 1],
        [l+w, 1, -1],
        [-l+w, 1, 1]
    ])

    velocity = np.linalg.pinv(H) @ u
    return velocity 

    return {
        'x': velocity[0],
        'y': velocity[1],
        'theta': velocity[2],
    }


def calc_velocity(velocity_amr, theta):
    return np.array([
        velocity_amr[0]*np.cos(theta) - velocity_amr[1]*np.sin(theta),
        velocity_amr[0]*np.sin(theta) + velocity_amr[1]*np.cos(theta),
        velocity_amr[2],
    ])

    return {
        'x': velocity_amr['x']*np.cos(theta) - velocity_amr['y']*np.sin(theta),
        'y': velocity_amr['x']*np.sin(theta) + velocity_amr['y']*np.cos(theta),
        'theta': velocity_amr['theta']
    }
