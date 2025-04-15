# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

import torch, gc, os
from datetime import datetime
import time


ALGORITHM_OUTPUT_ROOT_FOLDER = "/algorithm_output"
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
ALGORITHM_OUTPUT = ALGORITHM_OUTPUT_ROOT_FOLDER + "/" + timestamp

IPD_DATASET_ROOT_DIR = "/ipd" # keep in sync with run_container.sh
POSE_REFINER_TOTAL_ITERATIONS = 5
DEBUG_LEVEL = 1
DEBUG_DIR = '//debug'

TARGET_OBJECT_IDS = [8, 19]

# Change these to accomodate lower GPU memory:
INPUT_IMG_SHORTER_SIDE_LENGTH_PIXELS = 500
LOW_GPU_MEMORY_MODE = True

set_logging_format(level=logging.WARN)
set_seed(0)

# Example of VS Code configuration (launch.json) for debugpy
# in order to attach to running Docker container
# on localhost network:
# 
#   {
#     "version": "0.2.0",
#     "configurations": [
#       {
#         "name": "Python: Remote Attach",
#         "type": "debugpy",
#         "request": "attach",
#         "connect": {
#           "host": "localhost",
#           "port": 5678
#         },
#         "pathMappings": [
#           {
#             "localRoot": "${workspaceFolder}",
#             "remoteRoot": "."
#           }
#         ]
#       }
#     ]
#   }
DEBUG_STEP_THROUGH = False
if DEBUG_STEP_THROUGH:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))



if __name__=='__main__':

    os.system(f'rm -rf {DEBUG_DIR}/* && mkdir -p {DEBUG_DIR}/track_vis {DEBUG_DIR}/ob_in_cam')

    if DEBUG_STEP_THROUGH:
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
        print("Client attached. Continuing...")

    #reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=500, zfar=np.inf)
    reader = IpdReader(root_folder=IPD_DATASET_ROOT_DIR, shorter_side=INPUT_IMG_SHORTER_SIDE_LENGTH_PIXELS)

    # get only one group and one camera for now:
    group_id = reader.enumerate_groups()[0]
    camera_id = 1

    start_time_overrall = time.perf_counter()
    elapsed_time_inference = 0
    total_inferences_made = 0

    logging.info("Instantiating PoseEstimator class...")
    poseEstimator = PoseEstimator(debug=DEBUG_LEVEL, est_refine_iter=POSE_REFINER_TOTAL_ITERATIONS, 
                                  debug_dir=DEBUG_DIR, low_gpu_mem_mode=LOW_GPU_MEMORY_MODE)
    logging.info("Finished instantiating PoseEstimator class...")

    for scene_id in reader.enumerate_scenes(group_id):
        logging.info(f"Start of scene: {scene_id}")
        color = reader.get_rgb_image(group_id, scene_id, camera_id)
        depth = reader.get_depth_image(group_id, scene_id, camera_id)

        for object_class_id, object_instance_ids in reader.enumerate_objects(group_id, scene_id, camera_id).items():

            if TARGET_OBJECT_IDS is not None and len(TARGET_OBJECT_IDS) > 0 and object_class_id not in TARGET_OBJECT_IDS:
                continue

            object_instance_ids = object_instance_ids[0 : min(2, len(object_instance_ids)) ] # use only a couple of instances for each object class, just to test drive
            for object_instance_id in object_instance_ids:
                mask = reader.get_visible_object_mask(group_id, scene_id, camera_id, object_instance_id)
                mesh = reader.get_object_mesh(object_class_id)
                K = reader.get_camera_intrinsics_K_matrix(group_id, scene_id, camera_id)
                start_time_inference = time.perf_counter()
                pose = poseEstimator.estimate(object_class_id=object_class_id, K=K, mesh=mesh, color=color, depth=depth, mask=mask)
                end_time_inference = time.perf_counter()
                this_inference_time = end_time_inference - start_time_inference
                elapsed_time_inference += this_inference_time
                total_inferences_made += 1
                logging.info(f"Pose inference done in {this_inference_time:.1f} seconds")

                # We need to clean-up memory between object pose inferences, 
                # otherwise CUDA Out-of-Memory errors will frequently occur (non-deterministic).
                if LOW_GPU_MEMORY_MODE:
                    torch.cuda.empty_cache()
                    gc.collect()

                if DEBUG_LEVEL>=1 and pose is not None:
                    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                    center_pose = pose@np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox, linewidth=1)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                    
                    # Upscale vis color image by a factor of 2 for human visualization purposes.
                    vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                    
                    if os.path.isdir(ALGORITHM_OUTPUT_ROOT_FOLDER):
                        group_id_path = ALGORITHM_OUTPUT + "/" + f"{group_id:06d}"
                        scene_id_path = group_id_path + "/" + f"{scene_id:06d}"
                        camera_id_path = scene_id_path + "/" + str(camera_id) # no leading zeros for camera id numeric value
                        obj_class_id_path = camera_id_path + "/" + f"{object_class_id:06d}"
                        os.makedirs(obj_class_id_path, exist_ok=True)
                        image_title = f"{object_instance_id:06d}"
                        cv2.imwrite(obj_class_id_path + "/" + image_title + ".png", vis[...,::-1])
                    else:
                        cv2.imshow(f"{(group_id, scene_id, camera_id)} - {(object_class_id, object_instance_id)}", vis[...,::-1])
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()

    end_time_overall = time.perf_counter()

    elapsed_time_overall = end_time_overall - start_time_overrall
    logging.info(f"Elapsed time overall: {elapsed_time_overall:.1f} seconds")

    if total_inferences_made > 0:
        logging.info(f"Avg. loop iter time (per object instance): {elapsed_time_overall/total_inferences_made:.1f} seconds")
        logging.info(f"Avg. FoundationPose inference time: {elapsed_time_inference/total_inferences_made:.1f} seconds")
