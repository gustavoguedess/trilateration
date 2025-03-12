import numpy as np 
import amr_params as amr

def calc_velocity_ground_truth(last_gt, gt, time_diff):
    if time_diff == 0:
        return [0, 0, 0]
    velocity = [
        (gt[0] - last_gt[0])/time_diff,
        (gt[1] - last_gt[1])/time_diff,
        (gt[2] - last_gt[2])/time_diff,
    ]

    return velocity

def calc_amr_velocity(enc1, enc2, enc3, enc4):
    l = amr.WHEELS_LENGTH
    w = amr.WHEELS_WIDTH
    r = amr.WHEELS_RADIUS
    
    enc = np.array([enc1, enc2, enc3, enc4]).T

    time_simtwo = 1/25 # 25 Hz
    u = enc*2*np.pi/(30*64*2*time_simtwo) # 30 is the gear ratio, 64 is the encoder resolution. 2 is a magical number that makes it work

    ########################################
    H = 1/r * np.array([
        [1, -1, -l+w],
        [1, 1, l+w],
        [1, -1, l+w],
        [1, 1, -l+w]
    ])

    H_pseudoinverse = np.linalg.pinv(H)
    # velocity = H_pseudoinverse @ u
    
    ####### Alternative calculation #######
    value = 1/(l+w)
    H_pseudoinverse_alternative = r/4 * np.array([
        [1, 1, 1, 1],
        [-1, 1, -1, 1],
        [-value, value, value, -value],
    ])
    velocity = H_pseudoinverse_alternative @ u

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
