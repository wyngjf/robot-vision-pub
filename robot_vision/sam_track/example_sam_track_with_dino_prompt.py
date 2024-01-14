from asyncio.subprocess import DEVNULL
import os
import cv2
from pathlib import Path
from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import click
from robot_utils import console
from robot_utils.py.filesystem import copy2, get_ordered_files, validate_path, create_path, get_ordered_subdirs, validate_file
from robot_vision.grounded_sam.grounded_sam import GroundedSAM
from robot_vision.grounded_sam.wrapper import MaskWrapper
from icecream import ic
from robot_utils.torch.torch_utils import init_torch
from robot_vision.utils.utils import get_default_checkpoints_path


def save_prediction(pred_mask: np.ndarray, file_name: Path):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(str(file_name))


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0

    return img_mask.astype(img.dtype)


def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker


def get_demo_image_files(root_path_demo):
    console.rule("[blue]Loading reference demo images")
    demo_image_path_rgb = root_path_demo + "/rgb"
    demo_image_path_rgb, _ = validate_path(demo_image_path_rgb)
    #demo_image_path_depth = root_path_demo / "depth"

    demo_image_list_rgb = get_ordered_files(demo_image_path_rgb)
    #demo_image_list_depth = get_ordered_files(demo_image_path_depth)
    console.log(len(demo_image_list_rgb))

    # self.occluded_obj = demo_info.occluded_obj
    # self.occluded_time = demo_info.ot
    return demo_image_list_rgb


def sam_track_with_dino_prompt(path, obj_prompt):
    #path = "/common/homes/students/jin/nerf_kvil/explore/test_dataset"
    #video_name = 'asym_pour_0'
    io_args = {
        #'input_video': f'{path}/raw/{video_name}.mp4',
        'output_mask_dir': f'{path}/track_result/{obj_prompt}_masks',  # save pred masks
        'output_video': f'{path}/track_result/{obj_prompt}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
        'output_gif': f'{path}/track_result/{obj_prompt}_seg.gif', # mask visualization
    }

    # choose good parameters in sam_args based on the first frame segmentation result
    # other arguments can be modified in model_args.py
    # note the object number limit is 255 by default, which requires < 10GB GPU memory with amp
    sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }
    # grounding_caption = obj_prompt
    # box_threshold = 0.25
    # text_threshold = 0.25

    output_dir = create_path(io_args['output_mask_dir'])
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    training_root = get_default_checkpoints_path() / "mask"
    pred_list = []
    #masked_pred_list = []
    device = init_torch(seed=2023, use_gpu=True)
    mask_model = MaskWrapper(training_root=training_root, object_list=[obj_prompt], device=device)
    segtracker = SegTracker(segtracker_args, sam_args, aot_args)
    segtracker.restart_tracker()
    
    #load from picture dir
    fig_file_list = get_demo_image_files(path)
    with torch.cuda.amp.autocast():
        for idx, frame_file in enumerate(fig_file_list):
            frame = cv2.imread(filename=str(frame_file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if idx == 0:
                height, width, _ = frame.shape
                print("Detect")
                #predicted_mask, annotated_frame= segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
                #pred_dict = grounded_sam.run(frame)
                mask_dict = mask_model.get_mask_of_obj_with_highest_confidence(obj=obj_prompt,image=frame,erode_radius=5)
                ic(mask_dict[obj_prompt].shape)
                #ic(predicted_mask.shape)
                mask = mask_dict[obj_prompt]
                segtracker.add_reference(frame, mask)
                torch.cuda.empty_cache()
                obj_ids = np.unique(mask)
                #obj_ids = obj_ids[obj_ids!=0]
                print("processed frame {}, obj_num {}".format(idx, len(obj_ids)), end='\n')
                init_res = draw_mask(frame, mask, id_countour=False)
                plt.figure(figsize=(10, 10))
                plt.axis('off')
                plt.imshow(init_res)
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.axis('off')
                plt.imshow(colorize_mask(mask))
                plt.show()
            else:
                mask = segtracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            save_prediction(mask, output_dir / f'{idx:>02d}.png')
            pred_list.append(mask)


            print("processed frame {}, obj_num {}".format(idx,segtracker.get_obj_num()), end='\r')
        print('\nfinished')

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15

    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    gif_list = []
    for idx,frame_file in enumerate(fig_file_list):

        frame = cv2.imread(filename=str(frame_file))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[idx]
        masked_frame = draw_mask(frame,pred_mask)
        gif_list.append(colorize_mask(pred_mask))
        # masked_frame = masked_pred_list[frame_idx]
        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(idx),end='\r')
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # ================================== load from video ============================================
    # cap = cv2.VideoCapture(io_args['input_video'])
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_idx = 0

    # segtracker = SegTracker(segtracker_args,sam_args,aot_args)
    # segtracker.restart_tracker()
    # with torch.cuda.amp.autocast():
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if frame_idx == 0:
    #             frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #             print("Detect")
    #             predicted_mask, annotated_frame= segtracker.detect_and_seg(frame, grounding_caption, box_threshold, text_threshold)
    #             segtracker.add_reference(frame, predicted_mask)
    #             #pred_mask = segtracker.seg(frame)
    #             torch.cuda.empty_cache()
    #             obj_ids = np.unique(predicted_mask)
    #             #obj_ids = obj_ids[obj_ids!=0]
    #             print("processed frame {}, obj_num {}".format(frame_idx,len(obj_ids)),end='\n')
    #             init_res = draw_mask(frame,predicted_mask,id_countour=False)
    #             plt.figure(figsize=(10,10))
    #             plt.axis('off')
    #             plt.imshow(init_res)
    #             plt.show()
    #             plt.figure(figsize=(10,10))
    #             plt.axis('off')
    #             plt.imshow(colorize_mask(predicted_mask))
    #             plt.show()
    #         else:
    #             predicted_mask = segtracker.track(frame,update_memory=True)
    #         torch.cuda.empty_cache()
    #         gc.collect()
    #         save_prediction(predicted_mask,output_dir,str(frame_idx)+'.png')
    #         # masked_frame = draw_mask(frame,pred_mask)
    #         # masked_pred_list.append(masked_frame)
    #         # plt.imshow(masked_frame)
    #         # plt.show()

    #         pred_list.append(predicted_mask)


    #         print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
    #         frame_idx += 1
    #     cap.release()
    #     print('\nfinished')



    # if io_args['input_video'][-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif io_args['input_video'][-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    # gif_list = []
    # frame_idx = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #     pred_mask = pred_list[frame_idx]
    #     masked_frame = draw_mask(frame,pred_mask)
    #     gif_list.append(colorize_mask(pred_mask))
    #     # masked_frame = masked_pred_list[frame_idx]
    #     masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
    #     out.write(masked_frame)
    #     print('frame {} writed'.format(frame_idx),end='\r')
    #     frame_idx += 1
    # out.release()
    # cap.release()
    # print("\n{} saved".format(io_args['output_video']))
    # print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], gif_list, duration=(50))
    print("{} saved".format(io_args['output_gif']))

    # manually release memory (after cuda out of memory)
    del segtracker
    torch.cuda.empty_cache()
    gc.collect()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",     "-p",    type=str,   help="the absolute path to the demo")
#@click.option("--vidoe_name",    "-n",   type=str,   help="the name of video e.g. -n asym_pour_0 for asym_pour_0.mp4")
@click.option("--obj_prompt",     "-o",    type=str,   help="the prompt for grounding dino e.g. -o 'cup.kettle' ")
def main(path, obj_prompt):
    sam_track_with_dino_prompt(path=path, obj_prompt=obj_prompt)


if __name__ == "__main__":
    main()
