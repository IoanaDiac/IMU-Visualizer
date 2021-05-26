import math
from math import sin, cos, tan, pi
import pandas as pd
from time import sleep

df = pd.read_csv('imu_data.csv')
print(df)


def get_acc(df, index):
    ax = df.at[index, 'Ax'] / 16384.0
    ay = df.at[index, 'Ay'] / 16384.0
    az = df.at[index, 'Az'] / 16384.0
    return [ax, ay, az]


def get_gyro(df, index):
    gx = df.at[index, 'Gx'] * math.pi / (180.0 * 131.0)
    gy = df.at[index, 'Gy'] * math.pi / (180.0 * 131.0)
    gz = df.at[index, 'Gz'] * math.pi / (180.0 * 131.0)
    return [gx, gy, gz]


def get_acc_angles(df, index):
    [ax, ay, az] = get_acc(df, index)
    phi = math.atan2(ay, math.sqrt(ax ** 2.0 + az ** 2.0))
    theta = math.atan2(-ax, math.sqrt(ay ** 2.0 + az ** 2.0))
    return [phi, theta]


df_ang = pd.DataFrame(columns=['rotx', 'roty', 'rotz'])
sleep_time = 0.01

# Filter coefficient
alpha = 0.4

# Complimentary filter estimates
phi_hat = 0.0
theta_hat = 0.0
zeta = 0.0

dt = 0.0
start_time = df.at[0, 't']

for index in range(1999):
    dt = df.at[index + 1, 't'] - start_time
    start_time = df.at[index + 1, 't']

    # Get estimated angles from raw accelerometer data
    [phi_hat_acc, theta_hat_acc] = get_acc_angles(df, index)

    # Get raw gyro data
    [p, q, r] = get_gyro(df, index)

    # Calculate Euler angle derivatives
    phi_dot = p + sin(phi_hat) * tan(theta_hat) * q + cos(phi_hat) * tan(theta_hat) * r
    theta_dot = cos(phi_hat) * q - sin(phi_hat) * r

    # Update complimentary filter
    phi_hat = (1 - alpha) * (phi_hat + dt * phi_dot) + alpha * phi_hat_acc
    theta_hat = (1 - alpha) * (theta_hat + dt * theta_dot) + alpha * theta_hat_acc
    zeta = zeta + dt * r

    df_ang.loc[index] = [phi_hat * 180 / pi, theta_hat * 180 / pi, zeta * 180 / pi]

print(df_ang)
