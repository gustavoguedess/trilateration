import numpy as np 
import functions.utils as utils

X = 0
Y = 1
THETA = 2

def init_state():
    global transiction_matrix       # A
    global error_covariance         # P
    global error_covariance_predict # P'
    global process_noise            # Q
    global measurement_noise        # R
    global observation_matrix       # H
    global identity_matrix          # I
    global control_matrix           # B

    # Define the transiction matrix A
    transiction_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the control matrix B
    control_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1]
    ])

    # Define the error covariance P
    error_covariance = np.identity(3)

    # Define the process noise Q
    process_noise = np.array([
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.01]
    ])

    # Define the measurement noise R
    measurement_noise = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]
    ])

    #TODO: calculate the observation matrix H
    # Define the observation matrix H
    observation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the identity matrix I
    identity_matrix = np.identity(3)

def predict(state, state_increment):
    global error_covariance_predict # P'

    # calc the jacobian matrix A
    jacobian_matrix = np.array([
        [1, 0, -state_increment[X]*np.sin(state[THETA])+state_increment[Y]*np.cos(state[THETA])],
        [0, 1, state_increment[X]*np.cos(state[THETA])-state_increment[Y]*np.sin(state[THETA])],
        [0, 0, 1]
    ])
       
    # Predict the next state
    # x' = A*x + B*u
    predict_state = jacobian_matrix @ state + control_matrix @ state_increment

    #TODO check if this like is necessary
    # predict_state[2] = utils.normalise_angle(predict_state[2]) # [0, 2pi]

    # Predict the next error covariance
    # P' = A*P*A.T + Q
    error_covariance_predict = jacobian_matrix @ error_covariance @ jacobian_matrix.T + process_noise

    return predict_state

def correct(predict_state, tril_state):
    global error_covariance         # P
    
    # Calculate the Kalman Gain K
    # K = P' * H.T * (H * P' * H.T + R)^-1
    kalman_gain = error_covariance_predict @ observation_matrix.T @ np.linalg.inv(observation_matrix @ error_covariance_predict @ observation_matrix.T + measurement_noise)
    
    # Update estimate
    # x = x' + K * (z - H * x)
    state = predict_state + kalman_gain @ ( tril_state - observation_matrix @ predict_state)

    # Update the error covariance
    # P = (I - K * H) * P'
    error_covariance = (identity_matrix - kalman_gain @ observation_matrix) @ error_covariance_predict
    
    return state
