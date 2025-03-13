from functions.simtwo import SimTwo
import trilateration as tril
import time
import odometry
import ekf
import numpy as np

X = 0
Y = 1
THETA = 2

st = SimTwo()

def get_initial_trilateration():
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
    state = get_initial_trilateration()
    last_gt = state.copy()
    ekf.init_state()

    acm_time = 0
    while input_data := st.get_data():
        time_diff = get_diff_time()
        acm_time+=time_diff

        # Get data
        data_lidar = input_data['raw_lidar']
        ground_truth = st.state2array(input_data['ground_truth'])
        encoders = st.encoders2array(input_data['ground_truth'])

        # Calculate relative velocity of AMR
        velocity_amr = odometry.calc_amr_velocity(encoders = encoders)

        # Calculate absolute velocity of AMR
        velocity = odometry.calc_velocity(velocity_amr, ground_truth[THETA])

        # Calculate ground truth velocity
        gt_velocity = odometry.calc_velocity_ground_truth(last_gt, ground_truth, time_diff)
        last_gt = ground_truth
        
        # Trilateration
        tril_state = tril.calc_trilateration(data_lidar, last_state=state)

        # EKF Predict
        predict_state = ekf.predict(
            state = state,
            delta_state = velocity*time_diff,
        )

        # if acm_time > 0.05:
        #     state = ekf.correct (
        #         predict_state = predict_state,
        #         tril_state = tril_state,
        #     )
        #     acm_time = 0
        # else:
        #     state = predict_state

        # Print data to DEBUG
        print_time_diff(time_diff)
        print_encoders(input_data['ground_truth'])
        print_velocity(velocity_amr, 'AMR')
        print_velocity(velocity, 'Abs')
        print_velocity(gt_velocity, 'GT ')
        print_velocity_error(velocity, gt_velocity)
        print()
        
        print_state(tril_state, 'Trilateration')
        print_state(predict_state, 'Predicted')
        print_state(ground_truth, 'Ground truth')
        print_state_error(state, ground_truth)
        print()

        print_state(ground_truth, ' Ground truth')
        print_state(predict_state, '    Predicted')
        print_state(state, 'Trilateration')

        print_clear_last_line(n=20)



def print_state(state, name=''):
    x = state[X]
    y = state[Y]
    theta = state[THETA]

    if name:
        print(f'{name} ', end='')
    print(f'State: x={x:.4f}, y={y:.4f}, theta={theta:.2f}')

def print_state_error(state, ground_truth, name=''):
    x_error = abs(state[X] - ground_truth[X])
    y_error = abs(state[Y] - ground_truth[Y])
    theta_error = abs(state[THETA] - ground_truth[THETA])

    if name:
        print(f'{name} ', end='')
    print(f'Error: x={x_error:.4f}, y={y_error:.4f}, theta={theta_error:.2f}')

def print_velocity(velocity, name='', ground_truth=None):
    x, y, theta = velocity

    if name:
        print(f'{name} ', end='')
    print(f'Velocity: x={x: .4f}, y={y: .4f}, theta={theta: .2f}')

def print_velocity_error(velocity, ground_truth):
    x_error = abs(velocity[0] - ground_truth[0])
    y_error = abs(velocity[1] - ground_truth[1])
    theta_error = (velocity[2] - ground_truth[2] + np.pi) % (2*np.pi) - np.pi
    degree = np.rad2deg(theta_error)
    print(f'Velocity Error: x={x_error: .4f}, y={y_error: .4f}, theta={theta_error: .2f} ({degree: .2f}°)')

def print_encoders(ground_truth):
    u1 = ground_truth['u1']
    u2 = ground_truth['u2']
    u3 = ground_truth['u3']
    u4 = ground_truth['u4']

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
