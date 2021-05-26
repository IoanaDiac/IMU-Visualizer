from complementary import get_acc, get_acc_angles, get_gyro
import numpy as np
from time import sleep, time
from math import sin, cos, tan, pi
import pandas as pd

df = pd.read_csv('imu_data.csv')
print(df)


df_kalman = pd.DataFrame(columns=['rotx', 'roty', 'rotz'])
# Initialise matrices and variables
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
P = np.eye(4)
Q = np.eye(4)
R = np.eye(2)

state_estimate = np.array([[0], [0], [0], [0]])

phi_hat = 0.0
theta_hat = 0.0
zeta = 0.0

dt = 0.0
start_time = df.at[0, 't']

for index in range(1999):
    # Sampling time
    dt = df.at[index + 1, 't'] - start_time
    start_time = df.at[index + 1, 't']

    # Get accelerometer measurements and remove offsets
    [phi_acc, theta_acc] = get_acc_angles(df, index)

    # Gey gyro measurements and calculate Euler angle derivatives
    [p, q, r] = get_gyro(df, index)
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r

    # Kalman filter
    A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
    B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])

    gyro_input = np.array([[phi_dot], [theta_dot]])
    state_estimate = A.dot(state_estimate) + B.dot(gyro_input)
    P = A.dot(P.dot(np.transpose(A))) + Q

    measurement = np.array([[phi_acc], [theta_acc]])
    y_tilde = measurement - C.dot(state_estimate)
    S = R + C.dot(P.dot(np.transpose(C)))
    K = P.dot(np.transpose(C).dot(np.linalg.inv(S)))
    state_estimate = state_estimate + K.dot(y_tilde)
    P = (np.eye(4) - K.dot(C)).dot(P)

    phi_hat = float(state_estimate[0])
    theta_hat = float(state_estimate[2])
    zeta = zeta + dt * r


    df_kalman.loc[index] = [phi_hat * 180 / pi, theta_hat * 180 / pi, zeta * 180 / pi]

