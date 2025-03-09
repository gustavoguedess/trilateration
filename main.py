from functions.simtwo import SimTwo
import trilateration as tril
import time
import odometry

st = SimTwo()

def get_initial_trilateration():
    data = st.get_data()

    print('Data: ', data)

    x, y, theta, u1, u2, u3, u4 = data['ground_truth'].values()
    trilateration = {
        'x': x,
        'y': y,
        'theta': theta,
    }

    return trilateration

def main():
    trilateration = get_initial_trilateration()
    
    print('Trilateration: ', trilateration)

    time_last_iteration = time.time()
    while input_data := st.get_data():
        time_current_iteration = time.time()
        time_diff = time_current_iteration - time_last_iteration
        print('Time difference: ', time_diff)

        velocity = odometry.calc_velocity(
            time = time_diff,
            u1 = input_data['ground_truth']['u1'],
            u2 = input_data['ground_truth']['u2'],
            u3 = input_data['ground_truth']['u3'],
            u4 = input_data['ground_truth']['u4'],
        )

        trilateration = tril.calc_trilateration(input_data['raw_lidar'], last_trilateration=trilateration)

        # print_trilateration_error(trilateration, input_data['ground_truth'])

        time_last_iteration = time_current_iteration

def print_trilateration_error(trilateration, ground_truth):
    x_error = abs(trilateration['x'] - ground_truth['x'])
    y_error = abs(trilateration['y'] - ground_truth['y'])
    theta_error = abs(trilateration['theta'] - ground_truth['theta'])

    print(f'Error: x={x_error:.4f}, y={y_error:.4f}, theta={theta_error:.2f}')
    

if __name__ == '__main__':
    main()
