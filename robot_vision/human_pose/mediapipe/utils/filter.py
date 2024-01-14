import numpy as np
import mediapipe as mp

from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_kalman_results(image: np.ndarray, kalman_results: np.ndarray):
    """
    Draw one hand on the image

    Args:
        image: rgb image in cv2 format
        kalman_results: (21, 2) 2D hand landmark

    Returns: rgb image with hand skeleton overlay

    """
    results = kalman_results[:21]
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=r[0], y=r[1], z=0) for r in results
    ])

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return image
