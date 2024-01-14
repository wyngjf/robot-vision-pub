import cv2
import numpy as np
from robot_vision.human_pose.mediapipe.mp_hand_kalman import HandDetector
from robot_vision.human_pose.mediapipe.holistic_handedness_detection import handedness_detection
from robot_vision.human_pose.graphormer.get_joints_from_mesh import MeshGraphormerWrapper


def main():
    hand_detector = HandDetector()
    graphormer = MeshGraphormerWrapper()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        humans, viz_img = handedness_detection([frame], viz=True)

        # TODO check n_hands == 2 * n_humans
        hand_com = self.d.human.mp_hand_landmarks[humans["time_idx"]].mean(axis=-2)  # (n_hand, 2)
        # TODO update the above function to output multiple humans
        n_hands = self.d.human.n_hands
        handedness_list = [None] * n_hands
        included_idx = list(np.arange(n_hands))
        for idx, human in enumerate(humans["human_list"]):
            h = HumanProperty()
            dist_left = np.linalg.norm(hand_com[included_idx] - human["wrist_uv_left"], axis=-1)
            dist_right = np.linalg.norm(hand_com[included_idx] - human["wrist_uv_right"], axis=-1)
            h.left_hand_idx = included_idx[np.argmin(dist_left)]
            h.right_hand_idx = included_idx[np.argmax(dist_right)]
            handedness_list[h.left_hand_idx] = "left"
            handedness_list[h.right_hand_idx] = "right"

            included_idx.pop(included_idx.index(h.left_hand_idx))
            included_idx.pop(included_idx.index(h.right_hand_idx))
            self.d.human.human_id.humans.append(h)

        img_size_wh = frame.shape[:2][::-1]
        hand_mean_uv = (mp_uv.mean(axis=-2) * img_size_wh).astype(int)  # (T, n_hands, 2)
        hand_bbox_top_left = np.clip(
            hand_mean_uv - 112, np.zeros(2, dtype=int),
            np.array([img_size_wh[0] - 224, img_size_wh[1] - 224], dtype=int)
        )
        hand_bbox_bottom_right = hand_bbox_top_left + 224
        crop_bbox = np.concatenate((hand_bbox_top_left, hand_bbox_bottom_right), axis=-1)

        graphormer.detect_hand_on_image_batch()

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

        cv2.imshow("mediapipe", mp_img)