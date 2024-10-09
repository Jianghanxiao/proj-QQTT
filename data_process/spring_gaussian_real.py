import open3d as o3d
import numpy as np
import json

# Read the images
def reform_data(object, data_path):
    # Read the configs
    with open(f"{data_path}/dynamic/sequences/{object}/0.json", "r") as f:
        config = json.load(f)
    hit_frame = config["hit_frame"]
    config.pop("hit_frame")
    cameras = list(config.keys())
    for camera in cameras:
        img_naems = config[camera]
        for img_name in img_naems:
            img = o3d.io.read_image(f"{data_path}/dynamic/videos_images/{camera}/{img_name}")
            img = np.asarray(img)
            # import pdb
            # pdb.set_trace()
            img = o3d.geometry.Image(img)
            o3d.visualization.draw_geometries([img])
            import pdb
            pdb.set_trace()
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    object = "burger"
    data_path = f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/spring_gaussian/real_capture"
    # Collect the annotations into my customized format
    reform_data(object, data_path)
    