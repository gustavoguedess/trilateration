from functions.simtwo import SimTwo
import trilateration as tril
import time
import odometry
import ekf

st = SimTwo()

def get_initial_trilateration():
    data = st.get_data()

    x, y, theta, u1, u2, u3, u4 = data['ground_truth'].values()
    trilateration = {
        'x': x,
        'y': y,
        'theta': theta,
    }

    return trilateration

time_last_iteration = time.time()
def get_diff_time():
    global time_last_iteration
    time_current_iteration = time.time()
    time_diff = time_current_iteration - time_last_iteration
    time_last_iteration = time_current_iteration

    return time_diff

def main():
    state = get_initial_trilateration()
    ekf.init_state(state['x'], state['y'], state['theta'])
    last_gt = st.get_data()['ground_truth']
    last_gt = [last_gt['x'], last_gt['y'], last_gt['theta']]

    print_state(state, 'First ')
    print()

    acm_time = 0
    while input_data := st.get_data():
        time_diff = get_diff_time()
        acm_time+=time_diff

        velocity_amr = odometry.calc_amr_velocity(
            enc1 = input_data['ground_truth']['u1'],
            enc2 = input_data['ground_truth']['u2'],
            enc3 = input_data['ground_truth']['u3'],
            enc4 = input_data['ground_truth']['u4'],
        )
        velocity = odometry.calc_velocity(velocity_amr, input_data['ground_truth']['theta'])

        gt = [input_data['ground_truth']['x'], input_data['ground_truth']['y'], input_data['ground_truth']['theta']]
        gt_velocity = odometry.calc_velocity_ground_truth(last_gt, gt, time_diff)
        last_gt = gt

        # EKF Predict
        predict_state = ekf.predict(
            state = state,
            state_variation = velocity*time_diff,
        )

        tril_state = tril.calc_trilateration(input_data['raw_lidar'], last_trilateration=state)

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
        print_state_error(state, input_data['ground_truth'], 'Trilateration')
        print_encoders(input_data['ground_truth'])
        print_velocity(velocity_amr, 'AMR')
        print_velocity(velocity, 'Abs', ground_truth=gt_velocity)

        print()

        print_state(input_data['ground_truth'], ' Ground truth')
        print_state(predict_state, '    Predicted')
        print_state(state, 'Trilateration')

        print_clear_last_line(n=9)



def print_state(State, name=''):
    x = State['x']
    y = State['y']
    theta = State['theta']

    if name:
        print(f'{name} ', end='')
    print(f'State: x={x:.4f}, y={y:.4f}, theta={theta:.2f}')

def print_state_error(state, ground_truth, name=''):
    x_error = abs(state['x'] - ground_truth['x'])
    y_error = abs(state['y'] - ground_truth['y'])
    theta_error = abs(state['theta'] - ground_truth['theta'])

    if name:
        print(f'{name} ', end='')
    print(f'Error: x={x_error:.4f}, y={y_error:.4f}, theta={theta_error:.2f}')

def print_velocity(velocity, name='', ground_truth=None):
    x, y, theta = velocity

    if name:
        print(f'{name} ', end='')
    print(f'Velocity: x={x: .4f}, y={y: .4f}, theta={theta: .2f}', end='')

    if ground_truth:
        x_error = abs(velocity[0] - ground_truth[0])
        y_error = abs(velocity[1] - ground_truth[1])
        theta_error = abs(velocity[2] - ground_truth[2])
        print(f'    Error: x={x_error:.4f}, y={y_error:.4f}, theta={theta_error:.2f}', end='')
    print()

def print_encoders(ground_truth):
    u1 = ground_truth['u1']
    u2 = ground_truth['u2']
    u3 = ground_truth['u3']
    u4 = ground_truth['u4']

    print(f'Encoders: u1={u1:4.0f}    u2={u2:4.0f}    u3={u3:4.0f}    u4={u4:4.0f}')

def print_time_diff(time_diff):
    print(f'Time diff: {time_diff:.4f}s')

def print_clear_last_line(n=1):
    for _ in range(n):
        print('\033[F', end='') # Move cursor up one line
        print('\033[K', end='') # Clear line

if __name__ == '__main__':
    main()
