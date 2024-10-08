from qqtt import SpringMassSystem
import open3d as o3d
import numpy as np
import torch
from qqtt.utils import visualize_pc, cfg
import time

spring_index = []

def get_spring_mass_from_pcd(pcd, raidus=0.1, max_neighbours=20, device="cuda"):
    global spring_index
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    spring_flags = np.zeros((len(points), len(points)))
    springs = []
    rest_lengths = []
    spring_Y = []
    vertices = points  # Use the points as the vertices of the springs
    for i in range(len(vertices)):
        [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
            points[i], raidus, max_neighbours
        )
        # import pdb
        # pdb.set_trace()
        idx = idx[1:]
        for j in idx:
            if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])
                # Manually set two different parameters for the table case
                if points[i][1] < 0.49:
                    spring_index.append(0)
                else:
                    spring_index.append(1)
                # Manually set two different k for the table
                # if points[i][1] < 0.49:
                #     spring_Y.append(1e4)
                # else:
                spring_Y.append(1e6)
                rest_lengths.append(np.linalg.norm(points[i] - points[j]))
    springs = np.array(springs)
    rest_lengths = np.array(rest_lengths)
    spring_Y = np.array(spring_Y)
    masses = np.ones(len(vertices))
    return (
        torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
        torch.tensor(springs, dtype=torch.int32, device=cfg.device),
        torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
        torch.tensor(masses, dtype=torch.float32, device=cfg.device),
        torch.tensor(spring_Y, dtype=torch.float32, device=cfg.device),
    )


def get_spring_mass_visual(
    vertices,
    springs,
    rest_lengths,
    spring_forces,
    spring_isbreak,
    force_visual_scale=200,
):
    # The factor is used to scale the force
    # Color the springs with force information
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    temp_springs = []
    line_colors = []
    for i in range(len(springs)):
        if spring_isbreak[i] == 0:
            temp_springs.append(springs[i])
            line_colors.append(
                np.array([0.0, 1.0, 0.0]) * spring_forces[i] / force_visual_scale
            )
    lineset.lines = o3d.utility.Vector2iVector(temp_springs)
    lineset.colors = o3d.utility.Vector3dVector(line_colors)

    # Color the vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.paint_uniform_color([1, 0, 0])

    visuals = [lineset, pcd]
    return visuals


def visualize(
    init_vertices,
    init_springs,
    init_rest_lengths,
    simulator,
    display,
    frame_len=200,
    save=False,
):
    if display == "offline":
        start = time.time()
        vertices = [init_vertices.cpu()]
        for i in range(frame_len):
            print("Step: ", i)
            print(time.time() - start)
            x, _, _, _ = simulator.step()
            vertices.append(x.cpu())
        print(time.time() - start)
        vertices = torch.stack(vertices, dim=0)
        visualize_pc(
            vertices,
            visualize=True,
        )
        if save:
            points_trajectories = vertices.cpu().numpy()
            return points_trajectories

    else:
        lineset, pcd = get_spring_mass_visual(
            init_vertices.cpu().numpy(),
            init_springs.cpu().numpy(),
            init_rest_lengths.cpu().numpy(),
            spring_forces=np.zeros(len(init_springs)),
            spring_isbreak=np.zeros(len(init_springs)),
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(lineset)
        vis.add_geometry(pcd)
        # Define ground plane vertices
        ground_vertices = np.array(
            [[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]]
        )

        # Define ground plane triangular faces
        ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])

        # Create Open3D mesh object
        ground_mesh = o3d.geometry.TriangleMesh()
        ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
        ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
        ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
        vis.add_geometry(ground_mesh)

        view_control = vis.get_view_control()
        view_control.set_front([-1, 0, 0.5])
        view_control.set_up([0, 0, 1])
        view_control.set_zoom(3)

        start = time.time()

        if save:
            points_trajectories = [init_vertices.cpu().numpy()]

        for i in range(frame_len):
            # print(i, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(time.time() - start)
            vertices, springs, rest_lengths, spring_forces = simulator.step()
            new_lineset, new_pcd = get_spring_mass_visual(
                vertices.cpu().numpy(),
                springs.cpu().numpy(),
                rest_lengths.cpu().numpy(),
                spring_forces.cpu().numpy(),
                spring_isbreak=np.zeros(len(init_springs)),
            )

            if save:
                points_trajectories.append(vertices.cpu().numpy())

            lineset.points = new_lineset.points
            lineset.lines = new_lineset.lines
            lineset.colors = new_lineset.colors
            pcd.points = new_pcd.points
            vis.update_geometry(lineset)
            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
        print(time.time() - start)
        vis.destroy_window()
        if save:
            points_trajectories = np.array(points_trajectories)
            return points_trajectories


def demo1():
    cfg.device = "cuda"
    display = "online"
    # Load the teddy into taichi and create a simple spring-mass system
    teddy = o3d.io.read_point_cloud(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data/teddy.ply"
    )
    teddy.translate([0, 0, 1])
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([teddy, coordinate])
    init_vertices, init_springs, init_rest_lengths, init_masses, spring_Y = (
        get_spring_mass_from_pcd(teddy)
    )

    with torch.no_grad():
        simulator = SpringMassSystem(
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            dt=5e-5,
            num_substeps=100,
            spring_Y=spring_Y,
            init_masks=torch.zeros(len(init_vertices), device=cfg.device),
            collide_elas=cfg.init_collide_elas,
            collide_fric=cfg.init_collide_fric,
            dashpot_damping=100,
            drag_damping=1,
        )

        visualize(init_vertices, init_springs, init_rest_lengths, simulator, display)


def demo2():
    global spring_index

    cfg.device = "cuda"
    display = "online"

    init_vertices = []
    init_springs = []
    init_rest_lengths = []
    init_masses = []
    spring_Y = []
    init_masks = []

    for i in range(1):
        teddy = o3d.io.read_point_cloud(
            f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data/table.ply"
        )
        teddy.translate([0, 0.3 * i, 1 * (i + 1)])
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # o3d.visualization.draw_geometries([teddy, coordinate])
        init_vertices_, init_springs_, init_rest_lengths_, init_masses_, spring_Y_ = (
            get_spring_mass_from_pcd(teddy)
        )
        len_vert = len(init_vertices_)
        init_vertices.append(init_vertices_)
        init_springs.append(init_springs_ + i * len(init_vertices_))
        init_rest_lengths.append(init_rest_lengths_)
        init_masses.append(init_masses_)
        spring_Y.append(spring_Y_)
        init_masks.append(
            torch.ones(len_vert, device=cfg.device, dtype=torch.int32) * i
        )

    init_vertices = torch.cat(init_vertices, dim=0)
    init_springs = torch.cat(init_springs, dim=0)
    init_rest_lengths = torch.cat(init_rest_lengths, dim=0)
    init_masses = torch.cat(init_masses, dim=0)
    spring_Y = torch.cat(spring_Y, dim=0)
    init_masks = torch.cat(init_masks, dim=0)

    with torch.no_grad():
        simulator = SpringMassSystem(
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            dt=5e-5,
            num_substeps=100,
            spring_Y=3e4,
            init_masks=init_masks,
            collide_elas=cfg.init_collide_elas,
            collide_fric=cfg.init_collide_fric,
            dashpot_damping=100,
            drag_damping=1,
            spring_index=torch.tensor(
                spring_index, dtype=torch.int32, device=cfg.device
            ),
        )

        visualize(
            init_vertices,
            init_springs,
            init_rest_lengths,
            simulator,
            display,
            frame_len=200,
        )


def generate_data_billiard():
    cfg.device = "cuda"
    display = "online"

    radius = 0.1
    spacing = 0.02

    def generate_solid_sphere(radius, spacing):
        x = np.arange(-radius, radius + spacing, spacing)
        y = np.arange(-radius, radius + spacing, spacing)
        z = np.arange(-radius, radius + spacing, spacing)
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
        points = points[np.linalg.norm(points, axis=1) <= radius]
        return points

    solid_sphere_points = generate_solid_sphere(radius, spacing)
    solid_sphere_points[:, 2] += radius 

    init_vertices = []
    init_springs = []
    init_rest_lengths = []
    init_masses = []
    init_masks = []
    init_velocities = []

    pose = torch.tensor(
        [
            [-1, 0, 0],
            [-(1 + 2 * radius), radius, 0],
            [-(1 + 2 * radius), -radius, 0],
            [-(1 + 4 * radius), 0, 0],
            [-(1 + 4 * radius), 2 * radius, 0],
            [-(1 + 4 * radius), -2 * radius, 0],
            [1, 0, 0],
        ],
        device=cfg.device,
        dtype=torch.float32,
    )

    for i in range(len(pose)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.copy(solid_sphere_points))
        init_vertices_, init_springs_, init_rest_lengths_, init_masses_, spring_Y_ = (
            get_spring_mass_from_pcd(pcd)
        )

        init_vertices_ += pose[i]
        len_vert = len(init_vertices_)
        if i == len(pose) - 1:
            init_velocities_ = torch.zeros(
                (len_vert, 3), device=cfg.device
            ) + torch.tensor([-30, 0, 0], device=cfg.device, dtype=torch.float32)
        else:
            init_velocities_ = torch.zeros((len_vert, 3), device=cfg.device)

        init_vertices.append(init_vertices_)
        init_springs.append(init_springs_ + i * len(init_vertices_))
        init_rest_lengths.append(init_rest_lengths_)
        init_masses.append(init_masses_)
        init_masks.append(
            torch.ones(len_vert, device=cfg.device, dtype=torch.int32) * i
        )
        init_velocities.append(init_velocities_)

    init_vertices = torch.cat(init_vertices, dim=0)
    init_springs = torch.cat(init_springs, dim=0)
    init_rest_lengths = torch.cat(init_rest_lengths, dim=0)
    init_masses = torch.cat(init_masses, dim=0)
    init_masks = torch.cat(init_masks, dim=0)
    init_velocities = torch.cat(init_velocities, dim=0)
    save_velocities = init_velocities.clone()

    with torch.no_grad():
        simulator = SpringMassSystem(
            init_vertices,
            init_springs,
            init_rest_lengths,
            init_masses,
            dt=5e-5,
            num_substeps=100,
            spring_Y=3e5,
            collide_elas=0.8,
            collide_fric=0.2,
            dashpot_damping=100,
            drag_damping=1,
            collide_object_elas=0.95,
            collide_object_fric=0.1,
            init_masks=init_masks,
            init_velocities=init_velocities,
        )

        points_trajectories = visualize(
            init_vertices,
            init_springs,
            init_rest_lengths,
            simulator,
            display,
            frame_len=50,
            save=True,
        )
        # np.save(f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard.npy", points_trajectories)
        # np.save(f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_mask.npy", init_masks.cpu().numpy())
        # np.save(f"/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/billiard_velocities.npy", save_velocities.cpu().numpy())


if __name__ == "__main__":
    # demo1()
    demo2()
    # generate_data_billiard()
