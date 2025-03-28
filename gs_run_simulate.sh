output_dir="./gaussian_splatting/output_dynamic"

# views=("00000" "00025" "00050" "00075" "00100" "00125")
# views=("0" "1" "2")
views=("0")

# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

# scenes=("cloth_blue_fold" "cloth_blue_lift" "cloth_pant_fold" "cloth_pant_lift" 
#         "cloth_red_fold" "cloth_red_lift" "cloth_shirt_fold" "cloth_shirt_lift" 
#         "cloth_skirt_1_fold" "cloth_skirt_1_lift" "cloth_skirt_2_fold")

scenes=("double_stretch_sloth")

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

for scene_name in "${scenes[@]}"; do

    python gs_render_dynamics.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ./gaussian_splatting/output/${scene_name}/${exp_name} \
        --name ${scene_name} \
        --white_background

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done