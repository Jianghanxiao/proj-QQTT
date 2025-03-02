output_dir="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output"
output_video_dir="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_video"
scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
        "double_stretch_sloth" "double_stretch_zebra"
        "rope_double_hand"
        "single_clift_cloth_1" "single_clift_cloth_3"
        "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
        "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
        "single_push_rope" "single_push_rope_1" "single_push_rope_4"
        "single_push_sloth"
        "weird_package")

        
# scenes=("double_stretch_zebra")

##### Attempt 1: vanilla #####
# exp_name="vanilla"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'pcd'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 2: vanilla w/ mesh init #####
# exp_name="vanilla_init=mesh"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'mesh'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 3: vanilla w/ hybrid init #####
# exp_name="vanilla_init=hybrid"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 4: w/ non-isotropic gaussians #####
# exp_name="init=hybrid_iso=False_ldepth=0.0_lnormal=0.0_laniso_0.0_lseg=0.0"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --gs_init_opt 'hybrid'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 5: w/ non-isotropic gaussians + anisotropic loss #####
# exp_name="init=hybrid_iso=False_ldepth=0.0_lnormal=0.0_laniso_1.0_lseg=0.0"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 1.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --gs_init_opt 'hybrid'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 6: vanilla w/ high-res images #####
# exp_name="vanilla_use_high_res"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid' \
#         --use_high_res

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 7: w/ depth prior (1.0) #####
# exp_name="init=hybrid_iso=True_ldepth=1.0_lnormal=0.0_laniso_0.0_lseg=0.0"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 1.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 8: w/ seg prior (1.0) #####
# exp_name="init=hybrid_iso=True_ldepth=0.0_lnormal=0.0_laniso_0.0_lseg=1.0"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 1.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid'

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 9: w/ seg prior (1.0) + depth prior (0.001) #####
exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

# Iterate over each folder
for scene_name in "${scenes[@]}"; do
    echo "Processing: $scene_name"

    # Training
    python train.py \
        -s ../../gaussian_data/${scene_name} \
        -m ./output/${scene_name}/${exp_name} \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid'

    # Rendering
    python render.py \
        -s ../../gaussian_data/${scene_name} \
        -m ./output/${scene_name}/${exp_name} \

    # Convert images to video
    python ../../../utils-script/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
done


##### Attempt 10: w/ dense pts_per_triangles #####
# exp_name="vanilla_init=hybrid_pts_per_triangles=50"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid' \
#         --pts_per_triangles 50

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 11: w/ extreme dense pts_per_triangles #####
# exp_name="vanilla_init=hybrid_pts_per_triangles=100"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 0.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid' \
#         --pts_per_triangles 100

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done


##### Attempt 12: w/ seg prior (1.0) + depth prior (0.001) + disable_SH #####  --> not working
# exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0_disable_SH"

# # Iterate over each folder
# for scene_name in "${scenes[@]}"; do
#     echo "Processing: $scene_name"

#     # Training
#     python train.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --iterations 10000 \
#         --lambda_depth 0.001 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic 0.0 \
#         --lambda_seg 1.0 \
#         --use_masks \
#         --isotropic \
#         --gs_init_opt 'hybrid' \
#         --disable_sh

#     # Rendering
#     python render.py \
#         -s ../../gaussian_data/${scene_name} \
#         -m ./output/${scene_name}/${exp_name} \
#         --disable_sh

#     # Convert images to video
#     python ../../../utils-script/img2video.py \
#         --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
#         --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
# done





############################################################################################################
# ================== Old commands ==================
############################################################################################################


# folder_name="rope_double_hand_iter_10000_vanilla_iso_True"

# # training
# python train.py \
#     -s ../../data-3dgs/rope_double_hand \
#     -m ./output/${folder_name} \
#     --iterations 10000 \
#     --lambda_depth 0.0 \
#     --lambda_normal 0.0 \
#     --lambda_anisotropic 0.0 \
#     --use_masks \
#     --isotropic
#     # --mesh_path /home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/mesh-alignment/output/final_trimesh.glb
#     # --disable_sh \

# # --gs_path /home/haoyuyh3/Documents/maxhsu/qqtt/proj-QQTT/mesh-alignment/output/final_gs.ply

# # testing
# python render.py \
#     -s ../../data-3dgs/rope_double_hand \
#     -m ./output/${folder_name} \
#     # --disable_sh
#     # --remove_gaussians


# python ../../../utils-script/img2video.py \
#     --image_folder /home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output/${folder_name}/test/ours_10000/renders \
#     --video_path /home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_video/${folder_name}.mp4


# lambda_anisotropic_values=(1.0 0.1 0.01 0.001 0.0)

# for lambda_anisotropic in "${lambda_anisotropic_values[@]}"; do
#     # Training command with the current lambda anisotropic
#     python train.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_w_mask_lambda_iso="$lambda_anisotropic" \
#         --iterations 2000 \
#         --use_masks \
#         --lambda_depth 0.0 \
#         --lambda_normal 0.0 \
#         --lambda_anisotropic "$lambda_anisotropic"

#     # Testing command with the current lambda anisotropic
#     python render.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_w_mask_lambda_iso="$lambda_anisotropic" \
#         --remove_gaussians
# done



# lambda_depth_values=(5.0 1.0 0.1 0.01 0.001 0.0)

# for lambda_depth in "${lambda_depth_values[@]}"; do
#     # Training command with the current lambda depth
#     python train.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_lambda_depth="$lambda_depth" \
#         --iterations 2000 \
#         --lambda_depth "$lambda_depth"

#     # Testing command with the current lambda depth
#     python render.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_lambda_depth="$lambda_depth"
# done


# lambda_normal_values=(1.0 0.1 0.01 0.001 0.0)

# for lambda_normal in "${lambda_normal_values[@]}"; do
#     # Training command with the current lambda normal
#     python train.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_lambda_normal="$lambda_normal" \
#         --iterations 2000 \
#         --lambda_normal "$lambda_normal"

#     # Testing command with the current lambda normal
#     python render.py \
#         -s ../data-3dgs/rope_double_hand \
#         -m output/rope_double_hand_lambda_normal="$lambda_normal"
# done





# Test effectiveness of normal loss (this would have bugs when setting isotropic to True)
# python train.py \
#     -s ../data-3dgs/rope_double_hand \
#     -m "output/rope_double_hand_iso_False_iter_30000_normal_1.0" \
#     --iterations 30000 \
#     --lambda_depth 0.0 \
#     --lambda_normal 0.0 \
#     --use_masks

# python render.py \
#     -s ../data-3dgs/rope_double_hand \
#     -m "output/rope_double_hand_iso_False_iter_30000_normal_1.0" \
#     --remove_gaussians


# Test effectiveness of depth loss (this would have bugs when setting isotropic to True)
# python train.py \
#     -s ../data-3dgs/rope_double_hand \
#     -m "output/rope_double_hand_iso_False_iter_30000_depth_1.0" \
#     --iterations 30000 \
#     --lambda_depth 1.0 \
#     --lambda_normal 0.0 \
#     --use_masks

# python render.py \
#     -s ../data-3dgs/rope_double_hand \
#     -m "output/rope_double_hand_iso_False_iter_30000_depth_1.0" \
#     --remove_gaussians
