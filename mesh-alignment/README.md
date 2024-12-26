# Guideline on this mesh alignment code

## Generate 3D textured mesh from multi-view images (Use TRELLIS)
- Please follow [TRELLIS](https://github.com/microsoft/TRELLIS) repository and install the environment.
- Replace the original `example_multi_image.py` file with ours to get the following output.
    - generated 3D meshes named `sample.glb`.
    - corresponding gaussians named `sample.ply`.

## Align generated 3D mesh with the observation from RGB-D cameras
- Reference to the PhysGen-v2 implementation.
- Installation:
    - Python 3.8
    - PyTorch (`conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`)
    - Open3D 0.18.0
    - Trimesh 4.5.3
    - PyTorch3D 0.7.8 (`pip install "git+https://github.com/facebookresearch/pytorch3d.git"`)
- Prepare and setup all required file paths within `fit_object.py`, then execute it. Details of each parameter is illustrated in the code snippet.