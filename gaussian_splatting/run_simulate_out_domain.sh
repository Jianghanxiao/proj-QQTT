output_dir="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_out_domain"
output_dir_transfer="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_out_domain_transfer"

# views=("00000" "00025" "00050" "00075" "00100" "00125")
views=("0" "1" "2")

scenes=(
    "double_lift_cloth_1_to_single_clift_cloth_1"
    "double_lift_zebra_to_double_stretch_zebra"
    "single_clift_cloth_1_to_double_lift_cloth_1"
    "single_lift_cloth_1_to_single_clift_cloth_1"
    "single_push_rope_to_rope_double_hand"
    "double_lift_cloth_1_to_single_lift_cloth_1"
    "double_stretch_sloth_to_double_lift_sloth"
    "single_clift_cloth_1_to_single_lift_cloth_1"
    "single_lift_cloth_3_to_double_lift_cloth_3"
    "single_push_rope_to_single_lift_rope"
    "double_lift_cloth_3_to_single_clift_cloth_3"
    "double_stretch_zebra_to_double_lift_zebra"
    "single_clift_cloth_3_to_double_lift_cloth_3"
    "single_lift_cloth_3_to_single_clift_cloth_3"
    "double_lift_cloth_3_to_single_lift_cloth_3"
    "rope_double_hand_to_single_lift_rope"
    "single_clift_cloth_3_to_single_lift_cloth_3"
    "single_lift_rope_to_rope_double_hand"
    "double_lift_sloth_to_double_stretch_sloth"
    "rope_double_hand_to_single_push_rope"
    "single_lift_cloth_1_to_double_lift_cloth_1"
    "single_lift_rope_to_single_push_rope"
)
# scenes=("double_lift_cloth_1_to_single_clift_cloth_1")


for scene_name in "${scenes[@]}"; do

    # use target gaussians directly
    python render_dynamics_out_domain.py \
        -s ../../gaussian_data/double_lift_cloth_1 \
        -m ./output/double_lift_cloth_1/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \
        --exp_name ${scene_name} \
        --white_background

    for view_name in "${views[@]}"; do
        # Convert images to video
        python ../../../utils-script/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

    # transfer source gaussians to target gaussians
    # python render_dynamics_out_domain.py \
    #     -s ../../gaussian_data/double_lift_cloth_1 \
    #     -m ./output/double_lift_cloth_1/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \
    #     --exp_name ${scene_name} \
    #     --transfer_gs

    # for view_name in "${views[@]}"; do
    #     # Convert images to video
    #     python ../../../utils-script/img2video.py \
    #         --image_folder ${output_dir_transfer}/${scene_name}/${view_name} \
    #         --video_path ${output_dir_transfer}/${scene_name}/${view_name}.mp4
    # done

done