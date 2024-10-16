import open3d as o3d
import numpy as np
import json
import glob
from time import sleep


# Currently this doesn't consider different fx and fy
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        if not z_flip:
            shoot_points.append(
                np.dot(transformation, np.array([0, 0, -length, 1]))[0:3]
            )
        else:
            shoot_points.append(
                np.dot(transformation, np.array([0, 0, length, 1]))[0:3]
            )
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


# Get the point cloud from rgb and depth numpy array
def getPcdFromRgbd(
    rgb,
    depth,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    intrinsic=None,
    depth_scale=1,
    alpha_filter=False,
    opencv=False,
):
    # Make the rgb go to 0-1
    if rgb.max() > 1:
        rgb /= 255.0
    # Convert the unit to meter
    depth /= depth_scale

    height, width = np.shape(depth)
    points = []
    colors = []

    for y in range(height):
        for x in range(width):
            if alpha_filter:
                # Filter the background based on the alpha channel
                if rgb[y][x][3] != 1:
                    continue
            colors.append(rgb[y][x][:3])
            if fx != None:
                points.append(
                    [
                        (x - cx) * (depth[y][x] / fx),
                        (y - cy) * (depth[y][x] / fy),
                        depth[y][x],
                    ]
                )
            else:
                depth[y][x] *= -1
                old_point = np.array(
                    [(width - x) * depth[y][x], y * depth[y][x], depth[y][x]]
                )
                point = np.dot(np.linalg.inv(intrinsic), old_point)
                points.append(point[:3])

    # if opencv:
    #     points = np.array(points)
    #     points[:, 2:] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)[:, 0:3])

    return pcd


def getPCD(c2w, intrinsic, depth, data_path, object, camera, index):
    rgb = (
        np.asarray(
            o3d.io.read_image(f"{data_path}/{object}/imgs/{camera}/{index}.jpg"),
            dtype=np.float32,
        )
        / 255.0
    )
    depth = 1 / (depth + 0.1)
    pcd = getPcdFromRgbd(
        rgb,
        depth,
        # intrinsic=intrinsic,
        fx=intrinsic[0, 0],
        fy=intrinsic[1, 1],
        cx=intrinsic[0, 2],
        cy=intrinsic[1, 2],
        opencv=True,
    )
    pcd.transform(c2w)
    return pcd


if __name__ == "__main__":
    object = "burger"
    data_path = f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/reform_SG"

    # Load the camera parameters
    with open(f"{data_path}/{object}/camera_info.json", "r") as f:
        camera_info = json.load(f)

    intrinsic = np.array(camera_info["intrinsic"])
    c2ws = camera_info["c2ws"]
    cameras = list(c2ws.keys())

    # Get the frame number
    frame_num = len(glob.glob(f"{data_path}/{object}/imgs/{cameras[0]}/*.jpg"))

    # Load the depth data
    depths = {}
    index = 0
    for camera in cameras:
        depth = np.load(f"{data_path}/{object}/depths/{camera}.npz")["depth"]
        # depth = np.load(f"{data_path}/{object}/depths/whole.npz")["depth"]
        # depths[camera] = depth[index * 20 : index * 20 + 20]
        depths[camera] = depth
        index += 1

    total_pcds = []
    # Visualize the dynamic point cloud
    for camera in cameras:
        pcds = []
        for index in range(frame_num):
            pcd = getPCD(
                c2ws[camera],
                intrinsic,
                depths[camera][index],
                data_path,
                object,
                camera,
                index,
            )
            pcds.append(pcd)

        camera_vis = getCamera(
            c2ws[camera],
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
            scale=0.5,
            z_flip=True,
        )

        total_pcds += pcds
        total_pcds += camera_vis    
        # o3d.visualization.draw_geometries(pcds + camera_vis)

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=True)
        # # Get the point cloud
        # camera_vis = getCamera(
        #     c2ws[camera],
        #     intrinsic[0, 0],
        #     intrinsic[1, 1],
        #     intrinsic[0, 2],
        #     intrinsic[1, 2],
        #     scale=0.5,
        #     z_flip=True,
        # )
        # for camera_vis_item in camera_vis:
        #     vis.add_geometry(camera_vis_item)

        # vis.add_geometry(pcds[0])

        # for index in range(1, frame_num):
        #     print(index)
        #     sleep(1)
        #     pcd.points = o3d.utility.Vector3dVector(np.array(pcds[index].points))
        #     pcd.colors = o3d.utility.Vector3dVector(np.array(pcds[index].colors))
        #     vis.update_geometry(pcd)
        #     vis.poll_events()
        #     vis.update_renderer()

    coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    total_pcds.append(coordinates)
    o3d.visualization.draw_geometries(total_pcds)