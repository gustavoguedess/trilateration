import numpy as np 
import amr_params as amr

def calc_velocity_ground_truth(last_gt, gt, time_diff):
    velocity = np.array([
        (gt[0] - last_gt[0])/time_diff,
        (gt[1] - last_gt[1])/time_diff,
        (gt[2] - last_gt[2])/time_diff,
    ]).T

    return velocity

def calc_state_increment(encoders, angle, time_diff):
    amr_velocity = calc_amr_velocity(encoders)
    velocity = calc_velocity(amr_velocity, angle)

    state_increment = velocity*time_diff

    return state_increment


def calc_amr_velocity(encoders):
    l = amr.WHEELS_LENGTH
    w = amr.WHEELS_WIDTH
    r = amr.WHEELS_RADIUS

    time_simtwo = 1/25 # 25 Hz
    u = encoders*2*np.pi/(30*64*2*time_simtwo) # 30 is the gear ratio, 64 is the encoder resolution. 2 is a magical number that makes it work

    #### Calculate H PseudoInverse ####
    # H = 1/r * np.array([
    #     [1, -1, -l+w],
    #     [1, 1, l+w],
    #     [1, -1, l+w],
    #     [1, 1, -l+w]
    # ])
    # H_pseudoinverse = np.linalg.pinv(H)
    
    ######## H PseudoInverse #########
    value = 1/(l+w)
    H_pseudoinverse = r/4 * np.array([
        [1, 1, 1, 1],
        [-1, 1, -1, 1],
        [-value, value, value, -value],
    ])

    ##################################

    velocity = H_pseudoinverse @ u

    return velocity 

def calc_velocity(velocity_amr, theta):
    return np.array([
        velocity_amr[0]*np.cos(theta) - velocity_amr[1]*np.sin(theta),
        velocity_amr[0]*np.sin(theta) + velocity_amr[1]*np.cos(theta),
        velocity_amr[2],
    ]).T
