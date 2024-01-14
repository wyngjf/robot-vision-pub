import cv2
import copy
import numpy as np
from typing import Dict, Any, Tuple, List
from robot_utils import console

from robot_utils.cv.opencv import overlay_masks_on_image
from robot_vision.sam_track.SegTracker import SegTracker
from robot_vision.sam_track.model_args import aot_args, sam_args, segtracker_args


class SAMTracker:
    def __init__(self):
        self.tracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.reset()
        console.rule(f"[bold cyan]tracker started")

    def reset(self):
        self.tracker.reset()
        self.tracker.restart_tracker()
        self.step = 0
        self.n_masks = 0
        self.step = 0
        self.colors = None

    def track(
            self,
            image: np.ndarray,
            masks: List[np.ndarray],
            viz=False,
            erode_radius: List[int] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        if self.step == 0:
            self.tracker.add_masks(copy.deepcopy(image), masks)
            self.n_masks = len(masks) if isinstance(masks, list) else masks.shape[0]
            self.colors = [np.random.randint(0, 255, 3).astype(np.uint8) for _ in range(self.n_masks)]
        else:
            track_mask = self.tracker.track(frame=image, update_memory=True)
            masks = [track_mask == idx for idx in range(1, self.n_masks+1)]

        if erode_radius is not None:
            for i, r in enumerate(erode_radius):
                kernel = np.ones((r, r), np.uint8)
                masks[i] = cv2.erode(masks[i].astype(np.uint8), kernel, iterations=1)

        self.step += 1
        if viz:
            image = overlay_masks_on_image(image, masks, self.colors)
        return image, masks

