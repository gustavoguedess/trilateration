import numpy as np 

def init_state(x, y, theta):
    global state                    # x, y, theta
    global transiction_matrix       # A
    global error_covariance         # P
    global error_covariance_predict # P'
    global process_noise            # Q
    global measurement_noise        # R
    global observation_matrix       # H
    global identity_matrix          # I

    state = np.array([x, y, theta])

    # Define the transiction matrix A
    transiction_matrix = np.matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the error covariance P
    error_covariance = np.identity(3)

    # Define the process noise Q
    process_noise = np.matrix([
        [0.01, 0, 0],
        [0, 0.01, 0],
        [0, 0, 0.01]
    ])

    # Define the measurement noise R
    measurement_noise = np.matrix([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1]
    ])

    #TODO: calculate the observation matrix H
    # Define the observation matrix H
    observation_matrix = np.matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Define the identity matrix I
    identity_matrix = np.identity(3)

def predict(state, state_diff):
    global error_covariance
    global error_covariance_predict
    
    state = np.matrix([state['x'], state['y'], state['theta']]).T
    state_diff = np.matrix([state_diff[0], state_diff[1], state_diff[2]]).T

    #TODO: Check if this is correct.    
    # Predict the next state
    # x' = A*x + B*u
    predict_state = transiction_matrix @ state + state_diff

    # Predict the next error covariance
    # P' = A*P*A.T + Q
    error_covariance_predict = transiction_matrix @ error_covariance @ transiction_matrix.T + process_noise

    predict_state = {'x': predict_state[0, 0], 'y': predict_state[1, 0], 'theta': predict_state[2, 0]}
    return predict_state

def correct(predict_state, tril_state):
    global error_covariance         # P
    global error_covariance_predict # P'
    global measurement_noise        # R
    global observation_matrix       # H
    global identity_matrix          # I

    # Convert the states to numpy arrays
    predict_state = np.matrix([predict_state['x'], predict_state['y'], predict_state['theta']]).T
    tril_state = np.matrix([tril_state['x'], tril_state['y'], tril_state['theta']]).T
    
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
