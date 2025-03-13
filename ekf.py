import numpy as np 

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

def predict(state, delta_state):
    global error_covariance         # P
    global error_covariance_predict # P'
    global jacobian_matrix          # A

    # calc the jacobian matrix A
    jacobian_matrix = np.array([
        [1, 0, -delta_state[X]*np.sin(state[THETA])+delta_state[Y]*np.cos(state[THETA])],
        [0, 1, delta_state[X]*np.cos(state[THETA])-delta_state[Y]*np.sin(state[THETA])],
        [0, 0, 1]
    ])
       
    # Predict the next state
    # x' = A*x + B*u
    predict_state = jacobian_matrix @ state + control_matrix @ delta_state

    # Predict the next error covariance
    # P' = A*P*A.T + Q
    error_covariance_predict = transiction_matrix @ error_covariance @ transiction_matrix.T + process_noise

    return predict_state

def correct(predict_state, tril_state):
    global error_covariance         # P
    global error_covariance_predict # P'
    global measurement_noise        # R
    global identity_matrix          # I

    # Convert the states to numpy arrays
    predict_state = np.array([predict_state['x'], predict_state['y'], predict_state['theta']]).T
    tril_state = np.array([tril_state['x'], tril_state['y'], tril_state['theta']]).T
    
    # Calculate the Kalman Gain K
    # K = P' * H.T * (H * P' * H.T + R)^-1
    kalman_gain = error_covariance_predict @ observation_matrix.T @ np.linalg.inv(observation_matrix @ error_covariance_predict @ observation_matrix.T + measurement_noise)
    
    # breakpoint()
    #TODO: Check if this is correct. Should it be done without z? What does z even mean?
    # Update estimate
    # x = x + K * (- H * x)
    state = predict_state + kalman_gain @ (-observation_matrix @ predict_state)

    # Update the error covariance
    # P = (I - K * H) * P'
    error_covariance = (identity_matrix - kalman_gain @ observation_matrix) @ error_covariance_predict
    
    # Convert the state back to a dictionary
    state = {'x': state[0, 0], 'y': state[1, 0], 'theta': state[2, 0]}
    return state
