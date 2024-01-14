import os
import sys
import time
from pathlib import Path
from rich.progress import Progress
from robot_utils import console

NGP_PATH = os.environ.get("NGP_PATH")
if not NGP_PATH:
    console.rule("[bold red]missing DEX_NGP_PATH env variable")
    exit(1)
sys.path.append(str(Path(NGP_PATH, "scripts")))
from common import *
import pyngp as ngp


def train_ngp(
        scene_dir: Path,
        snapshot_file: Path = None,
        n_steps: int = 1000
):
    if snapshot_file is None or not snapshot_file.is_file():
        snapshot_file = scene_dir / "base.msgpack"
    base_network = Path(NGP_PATH) / "configs/nerf/base.json"

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    # testbed.nerf.sharpen = float(0)
    testbed.load_training_data(str(scene_dir))
    testbed.reload_network_from_file(str(base_network))
    testbed.shall_train = True
    testbed.nerf.render_with_camera_distortion = True
    # testbed.nerf.training.optimize_extrinsics = True
    # testbed.nerf.training.optimize_distortion = True

    if n_steps < 0:
        n_steps = 1000

    old_training_step = 0
    prev_time = 0
    with Progress() as p:
        p.console.rule("[bold blue]training NeRF")
        train_task = p.add_task("[blue]Training", total=n_steps)
        while testbed.frame():
            if testbed.want_repl():
                repl(testbed)

            if testbed.training_step >= n_steps:
                break

            now = time.monotonic()
            if now - prev_time > 0.1:
                p.update(train_task, advance=testbed.training_step - old_training_step)

                # t.update(testbed.training_step - old_training_step)
                # t.set_postfix(loss=testbed.loss)
                old_training_step = testbed.training_step
                prev_time = now

    if snapshot_file:
        console.print(f"Saving snapshot {snapshot_file}")
        testbed.save_snapshot(str(snapshot_file), False)

    testbed.clear_training_data()
    ngp.free_temporary_memory()

    console.rule("[bold blue]training finished")
