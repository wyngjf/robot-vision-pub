import numpy as np
from filterpy.kalman import KalmanFilter


def create_kalman_filter(dim_x: int, dim_z: int):
    """
    create kalman filter for the hand detection case
    Args:
        dim_x: state of the kalman filter
        dim_z: measurement dimension

    Returns: Kalman Filter instance

    """
    kalman_filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
    kalman_filter.x = np.zeros(dim_x)  # Initial state estimate
    kalman_filter.P *= 1000  # Initial state covariance
    kalman_filter.F = np.eye(dim_x)  # State transition matrix
    kalman_filter.H = np.eye(dim_x)  # Measurement matrix
    kalman_filter.R *= 0.01  # Measurement noise covariance
    kalman_filter.Q *= 0.01  # Process noise covariance
    return kalman_filter


def kalman_estimation(kalman_filter: KalmanFilter, measurement_array: np.ndarray = None) -> np.ndarray:
    """
    Update a single kalman filter, and get predictions from it
    Args:
        kalman_filter: the filter instance
        measurement_array: the detected array (measurement of the state) (N_keypoint, 2)

    Returns:

    """
    if measurement_array is not None:
        measurement = measurement_array.flatten()
        kalman_filter.predict()
        kalman_filter.update(measurement)
        return kalman_filter.x.reshape(-1, 2)
    else:
        kalman_filter.predict()
        return kalman_filter.x.reshape(-1, 2)
