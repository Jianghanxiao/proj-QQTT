output_dir="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_only_second"

# views=("00000" "00025" "00050" "00075" "00100" "00125")
views=("0" "1" "2")

scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
        "double_stretch_sloth" "double_stretch_zebra"
        "rope_double_hand"
        "single_clift_cloth_1" "single_clift_cloth_3"
        "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
        "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
        "single_push_rope" "single_push_rope_1" "single_push_rope_4"
        "single_push_sloth"
        "weird_package")


for scene_name in "${scenes[@]}"; do

    # use target gaussians directly
    python render_dynamics_only_second.py \
        -s ../../gaussian_data/double_lift_cloth_1 \
        -m ./output/double_lift_cloth_1/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \
        --name ${scene_name}
        # --white_background

    for view_name in "${views[@]}"; do
        # Convert images to video
        python ../../../utils-script/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done