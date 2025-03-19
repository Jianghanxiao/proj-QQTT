output_dir="/home/haoyuyh3/Documents/maxhsu/qqtt/gaussian-recon/gaussian-splatting/output_dynamic_gnn"

# views=("00000" "00025" "00050" "00075" "00100" "00125")
views=("0" "1" "2")

scenes=(
    "render-double_lift_cloth_1-model_50"
    "render-double_stretch_sloth-model_50"
    "render-single_clift_cloth_3-model_50"
    "render-single_lift_cloth-model_50"
    "render-single_lift_zebra-model_50"
    "render-single_push_sloth-model_50"
    "render-double_lift_cloth_3-model_50"
    "render-double_stretch_zebra-model_50"
    "render-single_lift_cloth_1-model_50"
    "render-single_lift_dinosor-model_50"
    "render-single_push_rope_1-model_50"
    "render-weird_package-model_50"
    "render-double_lift_sloth-model_50"
    "render-rope_double_hand-model_50"
    "render-single_lift_cloth_3-model_50"
    "render-single_lift_rope-model_50"
    "render-single_push_rope_4-model_50"
    "render-double_lift_zebra-model_50"
    "render-single_clift_cloth_1-model_50"
    "render-single_lift_cloth_4-model_50"
    "render-single_lift_sloth-model_50"
    "render-single_push_rope-model_50"
)
# scenes=("render-single_push_rope-model_50")


for scene_name in "${scenes[@]}"; do

    # use target gaussians directly
    python render_dynamics_gnn.py \
        -s ../../gaussian_data/double_lift_cloth_1 \
        -m ./output/double_lift_cloth_1/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \
        --exp_name ${scene_name}
        # --white_background

    for view_name in "${views[@]}"; do
        # Convert images to video
        python ../../../utils-script/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done