from __future__ import print_function
import numpy as np
import threading
from dataclasses import dataclass, astuple


@dataclass
class InitialState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0


@dataclass
class InitialStateCovariance:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0


@dataclass
class ProcessNoise:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0


@dataclass
class MeasurementNoise:
    distance: float = 0.0
    yaw: float = 0.0


@dataclass
class MeasurementNoiseOrientation:
    roll: float = 0.0
    pitch: float = 0.0


@dataclass
class EkfParams:
    initial_state: InitialState = InitialState()
    initial_state_covariance: InitialStateCovariance = InitialStateCovariance()
    process_noise: ProcessNoise = ProcessNoise()
    measurement_noise: MeasurementNoise = MeasurementNoise()
    measurement_noise_orientation: MeasurementNoiseOrientation = MeasurementNoiseOrientation(
    )

    dim_meas: int = 0
    dim_state: int = 0


class AcousticEkf(object):

    def __init__(self, ekf_params: EkfParams):
        self.ekf_params = ekf_params

        self._x_est_0 = np.array(astuple(
            self.ekf_params.initial_state)).reshape((-1, 1))
        self._p0_mat = self.ekf_params.initial_state_covariance
        self._p0_mat = np.array(
            np.diag([
                self.ekf_params.initial_state_covariance.x**2,
                self.ekf_params.initial_state_covariance.y**2,
                self.ekf_params.initial_state_covariance.z**2,
                self.ekf_params.initial_state_covariance.dx**2,
                self.ekf_params.initial_state_covariance.dy**2,
                self.ekf_params.initial_state_covariance.dz**2
            ]))
        self.w_mat = np.array([0.2**2]).reshape((-1, 1))
        self.v_mat = np.array(
            np.diag([
                self.ekf_params.process_noise.x**2,
                self.ekf_params.process_noise.y**2,
                self.ekf_params.process_noise.z**2,
                self.ekf_params.process_noise.dx**2,
                self.ekf_params.process_noise.dy**2,
                self.ekf_params.process_noise.dz**2
            ]))

        self._x_est = self._x_est_0
        self._x_est_last = self._x_est
        self._p_mat = self._p0_mat

        self._last_time_stamp_update = 0
        self._last_time_stamp_prediction = 0
        self.lock = threading.Lock()

    def get_x_est(self):
        return np.copy(self._x_est)

    def get_x_est_0(self):
        return np.copy(self._x_est_0)

    def get_p_mat(self):
        return np.copy(self._p_mat)

    def get_x_est_last(self):
        return np.copy(self._x_est_last)

    def reset(self, x_est_0=None, p0_mat=None):
        if x_est_0:
            self._x_est = x_est_0
        else:
            self._x_est = self._x_est_0
        if p0_mat:
            self._p_mat = p0_mat
        else:
            self._p_mat = self._p0_mat

    def predict(self, dt):
        self._x_est_last = self._x_est
        self._x_est = self.f_function(self.get_x_est(), dt)
        a_mat = self.get_f_jacobian(self.get_x_est(), dt)
        self._p_mat = np.matmul(np.matmul(a_mat, self.get_p_mat()),
                                a_mat.transpose()) + self.v_mat

        # reset EKF if unrealistic values
        # if np.absolute(self.get_x_est()[0]) > 10 or np.absolute(
        #         self.get_x_est()[1] > 10):
        #     print('Resetting EKF: x or y value too far outside tank')
        #     self.reset()
        # elif not np.all(np.isfinite(self.get_x_est())):
        #     print('Resetting EKF: unrealistically high value')
        #     print('x: ', self.get_x_est())
        #     self.reset()

        return True

    def update(self, measurement, anchor_position):
        # measurement is: distance to anchor position

        self._x_est_last = self._x_est
        z_est = self.h_function(self.get_x_est(), anchor_position)
        h_mat = self.get_h_jacobian(self.get_x_est(), anchor_position)

        y = measurement - z_est
        self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                self.get_p_mat(), y, h_mat,
                                                self.w_mat)

        return True

    def f_function(self, x_est, dt):
        x_next = np.copy(x_est)
        x_next[:3] = x_est + dt * x_est[3:]
        return x_next

    def get_f_jacobian(self, x_est, dt):
        A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        return A

    def h_function(self, x_est, anchor_position):
        z = self.get_distance(x_est, anchor_position)
        return z

    def get_h_jacobian(self, x_est, anchor_position):
        distance = self.get_distance(x_est, anchor_position)
        return np.array([(x_est[0] - anchor_position[0]) / distance,
                         (x_est[1] - -anchor_position[1]) / distance,
                         (x_est[2] - -anchor_position[2]) / distance, 0, 0,
                         0]).reshape((-1, 1))

    def get_distance(self, x_est, anchor_position):
        dist = np.sqrt((x_est[0] - anchor_position[0])**2 +
                       (x_est[1] - anchor_position[1])**2 +
                       (x_est[2] - anchor_position[2])**2)
        return dist

    def _update(self, x_est, p_mat, y, h_mat, w_mat):
        """ helper function for general update """

        # compute K gain
        tmp = np.matmul(np.matmul(h_mat, p_mat), h_mat.transpose()) + w_mat
        k_mat = np.matmul(np.matmul(p_mat, h_mat.transpose()),
                          np.linalg.inv(tmp))

        # update state
        x_est = x_est + np.matmul(k_mat, y)

        # update covariance
        p_tmp = np.eye(self.ekf_params.dim_state) - np.matmul(k_mat, h_mat)
        p_mat = np.matmul(p_tmp, p_mat)
        # print('P_m diag: ', np.diag(self.get_p_mat()))
        return x_est, p_mat