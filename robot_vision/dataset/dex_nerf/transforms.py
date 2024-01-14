"""
Adapted from the instant ngp repository
"""
import cv2
import json
import math
import numpy as np
from pathlib import Path
from typing import Literal, Union
from robot_utils import console


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image_path: Path):
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):
    """
    returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta < 0:
        ta = 0
    if tb < 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def generate_transforms(
        scene:              Union[str, Path],
        aabb_scale:         int = 4,
        skip_early:         int = 0,
        frame_convention:   Literal["opencv", "opengl", "nerf", "ngp"] = "opencv"
):
    scene = Path(scene)
    transform_file = scene / "transforms.json"

    image_dir = scene / "images"
    if not image_dir.exists():
        raise FileExistsError("please run ffmpeg to extract images")

    colmap_text = scene / "colmap/colmap_text"
    if not colmap_text.exists():
        raise FileExistsError("Please run colmap first")

    console.rule("[bold blue]Generating transforms.json")
    console.print(f"writing to {transform_file}")
    with open(colmap_text / "cameras.txt", "r") as f:
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    console.print(f"camera:\n\tres={w, h}\n\tcenter={cx, cy}\n\tfocal={fl_x, fl_y}\n\tfov={fovx, fovy}\n\tk={k1, k2} p={p1, p2} ")

    with open(colmap_text / "images.txt", "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": aabb_scale,
            "fovx": fovx,
            "fovy": fovy,
            "d_rotation": [],
            "d_offset": [],
            "d_scale": 1.0,
            "offset": [0.5, 0.5, 0.5],
            "scale": 0.33,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            # images with no 3D points
            if len(line) == 0:
                i = i + 1
                continue
            if line[0] == "#":
                continue
            i = i + 1
            if i < skip_early:
                continue
            if i % 2 == 1:
                # 1-4 is quaternion (x, y, z, w) format, 5-7 is translation,
                # 9ff is filename (9, if filename contains no spaces)
                elems = line.split(" ")
                # image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                cam_2_world = np.linalg.inv(m)

                name = str(f"images/{'_'.join(elems[9:])}")
                b = sharpness(scene / name)

                if frame_convention == "opencv":
                    # camera in the opencv convention, for a roughly 360 scene,
                    # the overall up direction can be estimated using the -z dir of all camera frames
                    up += -cam_2_world[0:3, 2]
                else:
                    raise NotImplementedError(f"frame convention {frame_convention} is not supported yet")

                frame = {"file_path": name, "sharpness": b, "transform_matrix": cam_2_world}
                out["frames"].append(frame)

    n_frames = len(out["frames"])
    console.print(f"use {n_frames} frames in total")
    if n_frames <= 50:
        key = console.input(f"[red bold]Are you OK with {n_frames} frames? Press Enter to continue, 'q' to quite")
        if key == 'q':

            exit(0)

    # estimate up dir
    up = up / np.linalg.norm(up)
    console.log("up vector was", up)
    up_rotation = np.identity(4, dtype=float)
    up_rotation[:3, :3] = rotmat(up, [0, 1, 0])  # rotate up vector to [0, 1, 0], since in ngp, y is up as default
    out["d_rotation"] = up_rotation.tolist()
    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(up_rotation, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
    totw = 0.0
    d_offset = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.00001:
                d_offset += p * w
                totw += w
    if totw > 0.0:
        d_offset /= totw
    out["d_offset"] = d_offset.tolist()
    console.log(f"center of attention {d_offset}")  # the cameras are looking at totp

    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= d_offset

    # Estimate the scaling factor based on the average distance of each cam frame to the center of attention
    avg_len = 0.
    for f in out["frames"]:
        avg_len += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avg_len /= n_frames
    scale = 4.0 / avg_len
    out["d_scale"] = 1. / scale
    console.print("avg camera distance from origin", avg_len)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= scale

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    with open(transform_file, "w") as outfile:
        json.dump(out, outfile, indent=2)

    console.rule("[bold blue]transforms.json is generated")
