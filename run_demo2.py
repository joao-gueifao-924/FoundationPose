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

import torch, gc
import time

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import debugpy
debugpy.listen(("0.0.0.0", 5678))

if __name__=='__main__':
    ipd_dataset_root_dir = "/ipd"
    est_refine_iter=5
    debug=1
    debug_dir='//debug'
    shorter_side = 500

    set_logging_format()
    set_seed(0)
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    print("Waiting for client to attach...")
    debugpy.wait_for_client()

    #reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=500, zfar=np.inf)
    reader = IpdReaderTRAIN(root_folder=ipd_dataset_root_dir, shorter_side=shorter_side)

    # get only one group and one camera for now:
    group_id = reader.enumerate_groups()[0]
    camera_id = 1

    start_time_overrall = time.perf_counter()
    elapsed_time_inference = 0
    total_inferences_made = 0

    print("Instantiating PoseEstimator class...")
    poseEstimator = PoseEstimator(debug=debug)
    print("Finished instantiating PoseEstimator class...")

    for scene_id in reader.enumerate_scenes(group_id):
        logging.info(f"Start of scene: {scene_id}")
        color = reader.get_rgb_image(group_id, scene_id, camera_id)
        depth = reader.get_depth_image(group_id, scene_id, camera_id)

        for object_class_id, object_instance_ids in reader.enumerate_objects(group_id, scene_id, camera_id).items():
            object_instance_ids = [object_instance_ids[0]] # use only one object class instance for now, just to test drive
            for object_instance_id in object_instance_ids:
                mask = reader.get_visible_object_mask(group_id, scene_id, camera_id, object_instance_id)
                mesh = reader.get_object_mesh(object_class_id)
                K = reader.get_camera_intrinsics_K_matrix(group_id, scene_id, camera_id)
                start_time_inference = time.perf_counter()
                pose = poseEstimator.estimate(K=K, mesh=mesh, color=color, depth=depth, mask=mask)
                end_time_inference = time.perf_counter()
                elapsed_time_inference += end_time_inference - start_time_inference
                total_inferences_made += 1
                logging.info("pose inference done")

                # We need to clean-up memory between object pose inferences, 
                # otherwise CUDA Out-of-Memory errors will frequently occur (non-deterministic).
                
                torch.cuda.empty_cache()
                gc.collect()
                # time.sleep(5)

                # for i in range(100):
                #     print("#", end='')

                # print("-------------", end = '\n')

                if debug>=1 and pose is not None:
                    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
                    center_pose = pose@np.linalg.inv(to_origin)
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox, linewidth=1)
                    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
                    
                    # Upscale vis color image by a factor of 2 for human visualization purposes.
                    vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
                    cv2.imshow(f"{(group_id, scene_id, camera_id)} - {(object_class_id, object_instance_id)}", vis[...,::-1])
                    cv2.waitKey(3000)
                    cv2.destroyAllWindows()
                    


    end_time_overall = time.perf_counter()

    elapsed_time_overall = end_time_overall - start_time_overrall
    print(f"Elapsed time overall: {elapsed_time_overall:.1f} seconds")

    if total_inferences_made > 0:
        print(f"Avg. loop iter time (per object instance): {elapsed_time_overall/total_inferences_made:.1f} seconds")
        print(f"Avg. FoundationPose inference time: {elapsed_time_inference/total_inferences_made:.1f} seconds")
