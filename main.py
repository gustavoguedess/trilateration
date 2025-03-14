from functions.simtwo import SimTwo
import trilateration as tril
import time
import odometry
import ekf
import numpy as np
import amr_params as amr
import functions.utils as utils

X = 0
Y = 1
THETA = 2

COLOUR_BROWN = '\033[0;33m'
COLOUR_CYAN = '\033[0;36m'
COLOUR_YELLOW = '\033[1;33m'
COLOUR_GREEN = '\033[0;32m'
COLOUR_RESET = '\033[0m'


st = SimTwo()

def get_initial_state():
    data = st.get_data()

    state = st.state2array(data['ground_truth'])
    encoders = st.encoders2array(data['ground_truth'])

    return state


time_last_iteration = time.time()
def get_diff_time():
    global time_last_iteration
    time_current_iteration = time.time()
    time_diff = time_current_iteration - time_last_iteration
    time_last_iteration = time_current_iteration

    return time_diff

def main():
    state = get_initial_state()
    last_gt = state.copy()
    ekf.init_state()

    tril_state = state.copy()
    ekf_state = state.copy()
    acm_state = state.copy()*0

    acm_time = 0
    while input_data := st.get_data():
        time_diff = get_diff_time()
        acm_time+=time_diff

        # Get data
        data_lidar = input_data['raw_lidar']
        encoders = st.encoders2array(input_data['ground_truth'])

        # # Calculate relative velocity of AMR
        # velocity_amr = odometry.calc_amr_velocity(encoders = encoders)
        
        # # Calculate absolute velocity of AMR
        # velocity = odometry.calc_velocity(velocity_amr, ground_truth[THETA])
        
        state_increment = odometry.calc_state_increment(
            encoders = encoders,
            angle = state[THETA],
            time_diff = time_diff
        )
        acm_state += state_increment

        # EKF Predict
        predict_state = ekf.predict(
            state = state,
            state_increment = state_increment,
        )

        # Sync with LIDAR
        if acm_time > amr.LIDAR_PERIOD and False:
            
            # Trilateration
            tril_state = tril.calc_trilateration(data_lidar, last_state=state)

            # EKF Correct
            ekf_state = ekf.correct (
                predict_state = predict_state,
                tril_state = tril_state,
            )
            state = ekf_state

            acm_time = 0
        else:
            state = predict_state

        ground_truth = st.state2array(input_data['ground_truth'])
        print_debug(time_diff, ground_truth, encoders, state_increment, acm_state, tril_state, predict_state, ekf_state, state)


def print_debug(time_diff, ground_truth, encoders, state_increment, acm_state, tril_state, predict_state, ekf_state, state):
    print_clear_last_line(n=30)
    # Print data to DEBUG
    print_time_diff(time_diff)
    print()
    print('AMR')
    print_encoders(encoders)
    print_state(state_increment, 'Increment    ')
    print_state(acm_state, 'Accumulated  ')
    print_state_error(acm_state, ground_truth, '      ')
    print()
    
    print('State')
    print_state(ground_truth, 'Ground truth ', ground_truth=True)
    print_state(tril_state, 'Trilateration')
    print_state_error(tril_state, ground_truth, '      ')
    print()
    print_state(predict_state, 'Predicted    ')
    print_state_error(predict_state, ground_truth, '      ')
    print()
    print_state(ekf_state, 'Corrected    ')
    print_state_error(ekf_state, ground_truth, '      ')
    print()
    print()

    print_state(state, '             ')
    print_state_error(state, ground_truth, '      ')
    print()




def print_state(state, name='', ground_truth=False):
    x = state[X]
    y = state[Y]
    theta = state[THETA]

    if ground_truth:
        print(f'{COLOUR_GREEN}', end='')
    if name:
        print(f'{name} ', end='')
    print(f'x={x: .4f}, y={y: .4f}, theta={theta: .2f} ({np.rad2deg(theta):7.2f}°){COLOUR_RESET}')


def print_state_error(state, ground_truth, name=''):
    x_error = abs(state[X] - ground_truth[X])
    y_error = abs(state[Y] - ground_truth[Y])
    theta_error = utils.min_angle_diff(state[THETA], ground_truth[THETA])

    print(f'{COLOUR_BROWN}', end='')
    if name:
        print(f'{name} ', end='')
    print(f' Error x={x_error: .4f}, y={y_error: .4f}, theta={theta_error: .2f} ({np.rad2deg(theta_error): >7.2f}°){COLOUR_RESET}')

def print_velocity(velocity, name='', ground_truth=None):
    x, y, theta = velocity

    if ground_truth:
        print(f"{COLOUR_GREEN}", end='')
    if name:
        print(f'{name} ', end='')
    print(f' x={x: .4f}, y={y: .4f}, theta={theta: .2f} ({np.rad2deg(theta):7.2f}°){COLOUR_RESET}')

def print_velocity_error(velocity, ground_truth):
    x_error = abs(velocity[X] - ground_truth[X])
    y_error = abs(velocity[Y] - ground_truth[Y])
    theta_error = utils.min_angle_diff(velocity[THETA], ground_truth[THETA])
    degree = np.rad2deg(theta_error)
    print(f'{COLOUR_BROWN}Error         x={x_error: .4f}, y={y_error: .4f}, theta={theta_error: .2f} ({degree:7.2f}°){COLOUR_RESET}')

def print_encoders(encoders):
    u1, u2, u3, u4 = encoders
    print(f'Encoders: u1={u1:4.0f}    u2={u2:4.0f}    u3={u3:4.0f}    u4={u4:4.0f}')

def print_time_diff(time_diff):
    print(f'Time diff: {time_diff:.4f}s')

def print_clear_screen():
    print('\033[2J', end='')

def print_clear_last_line(n=1):
    for _ in range(n):
        print('\033[F', end='') # Move cursor up one line
        print('\033[K', end='') # Clear line

if __name__ == '__main__':
    main()
