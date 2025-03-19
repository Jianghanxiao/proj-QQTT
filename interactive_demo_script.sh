# double_lift_cloth_1  double_lift_zebra     rope_double_hand      single_lift_cloth    single_lift_cloth_4  single_lift_sloth  single_push_rope_1  weird_package
# double_lift_cloth_3  double_stretch_sloth  single_clift_cloth_1  single_lift_cloth_1  single_lift_dinosor  single_lift_zebra  single_push_rope_4
# double_lift_sloth    double_stretch_zebra  single_clift_cloth_3  single_lift_cloth_3  single_lift_rope     single_push_rope   single_push_sloth

##### No gaussians removal version #####

##### Type 1: cloth #####

# double_lift_cloth_1 (color a bit off 3/5)
# python interactive_playground.py --n_ctrl_parts 2 --inv_ctrl --case_name double_lift_cloth_1

# (SELECTED) double_lift_cloth_3 (goood 4/5)
# python interactive_playground.py --n_ctrl_parts 2 --inv_ctrl --case_name double_lift_cloth_3

# (SELECTED) single_clift_cloth_1 (color a bit off, no big movement 2/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_clift_cloth_1

# single_clift_cloth_3 (goood, tough for single hand control 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_clift_cloth_3

# single_lift_cloth (seems too rigid, 2/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_lift_cloth

# single_lift_cloth_1 (transparent, hand sprite a bit off, 1/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_lift_cloth_1

# single_lift_cloth_3 (nice movement, hand sprite a bit off, 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_lift_cloth_3

# (SELECTED) single_lift_cloth_4 (transparent, unique appearance, 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_lift_cloth_4


##### Type 2: stuffed animals #####

# double_lift_sloth (head shrink 3/5)
# python interactive_playground.py --n_ctrl_parts 2 --case_name double_lift_sloth

# (SELECTED) double_lift_zebra (mostly good 4/5)
# python interactive_playground.py --n_ctrl_parts 2 --case_name double_lift_zebra

# (SELECTED) double_stretch_sloth (head intact 5/5)
# python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_sloth

# double_stretch_zebra (mostly good 4/5)
# python interactive_playground.py --n_ctrl_parts 2 --case_name double_stretch_zebra

# (SELECTED) single_lift_dinosor (floaters, unique appearance, nice to drag clockwise 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_lift_dinosor

# single_lift_sloth (not much floaters, head intact 4/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_lift_sloth

# single_lift_zebra (another hand seems weird 2/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_lift_zebra

# single_push_sloth (heads completely off 1/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_push_slot


##### Type 3: ropes #####

# (SELECTED) rope_double_hand (floaters 3/5)
# python interactive_playground.py --n_ctrl_parts 2 --case_name rope_double_hand

# (SELECTED) single_lift_rope (back side have black dots 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_lift_rope

# single_push_rope (cool but two sides tends to stick together 3/5)
# python interactive_playground.py --n_ctrl_parts 1 --case_name single_push_rope

# (SELECTED) single_push_rope_1 (nice movement and dynamics 4/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_push_rope_1

# single_push_rope_4 (dynamics may not match observation? 1/5)
# python interactive_playground.py --n_ctrl_parts 1 --inv_ctrl --case_name single_push_rope_4


##### Type 4: packages #####

# weird_package (extremely slow FPS 1/5) <-- take much time in simulation
# python interactive_playground.py --n_ctrl_parts 2 --case_name weird_package



##### Gaussians removal version #####
# --remove_gs_from_mesh
# --remove_dist_th 0.001