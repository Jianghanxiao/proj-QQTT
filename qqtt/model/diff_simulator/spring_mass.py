import torch
import torch.nn as nn


# Differentialable Spring-Mass Simulator
class SpringMassSystem(nn.Module):
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt=5e-5,
        num_substeps=1000,
        spring_Y=3e4,
        dashpot_damping=100,
        drag_damping=1,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        # Number of mass and springs
        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]
        # Initialization
        self.x = init_vertices
        self.v = torch.zeros((self.n_vertices, 3), device=self.device)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
        # Internal forces
        self.spring_forces = None
        self.vertice_forces = torch.zeros((self.n_vertices, 3), device=self.device)

        self.dt = dt
        self.num_substeps = num_substeps
        self.spring_Y = spring_Y
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

    def step(self):
        for i in range(self.num_substeps):
            self.substep()

        return (
            self.x,
            self.springs,
            self.rest_lengths,
            self.spring_forces,
        )

    def substep(self):
        # One simulation step of the spring-mass system
        self.vertice_forces.zero_()

        # Add teh gravity force
        self.vertice_forces += self.masses[:, None] * torch.tensor(
            [0.0, 0.0, -9.8], device=self.device
        )
        # Calculate the spring forces
        idx1 = self.springs[:, 0]
        idx2 = self.springs[:, 1]
        x1 = self.x[idx1]
        x2 = self.x[idx2]
        dis = x2 - x1
        d = dis / torch.norm(dis, dim=1)[:, None]
        self.spring_forces = (
            self.spring_Y
            * (torch.norm(dis, dim=1) / self.rest_lengths - 1)[:, None]
            * d
        )

        self.vertice_forces.index_add_(0, idx1, self.spring_forces)
        self.vertice_forces.index_add_(0, idx2, -self.spring_forces)

        # Apply the damping forces
        v_rel = torch.einsum("ij,ij->i", (self.v[idx2] - self.v[idx1]), d)
        dashpot_forces = self.dashpot_damping * v_rel[:, None] * d
        self.vertice_forces.index_add_(0, idx1, dashpot_forces)
        self.vertice_forces.index_add_(0, idx2, -dashpot_forces)

        # Update the velocity
        self.v += self.dt * self.vertice_forces / self.masses[:, None]
        self.v *= torch.exp(
            torch.tensor(-self.dt * self.drag_damping, device=self.device)
        )
        self.x += self.dt * self.v

        # Simple ground condition for now
        self.x[:, 2].clamp_(min=0)
        self.v[self.x[:, 2] == 0, 2] = 0


if __name__ == "__main__":
    import open3d as o3d
    import numpy as np

    def get_spring_mass_from_pcd(pcd, raidus=0.1, max_neighbours=20):
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        spring_flags = np.zeros((len(points), len(points)))
        springs = []
        rest_lengths = []
        vertices = points  # Use the points as the vertices of the springs
        for i in range(len(vertices)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                points[i], raidus, max_neighbours
            )
            idx = idx[1:]
            for j in idx:
                if spring_flags[i, j] == 0 and spring_flags[j, i] == 0:
                    spring_flags[i, j] = 1
                    spring_flags[j, i] = 1
                    springs.append([i, j])
                    rest_lengths.append(np.linalg.norm(points[i] - points[j]))
        springs = np.array(springs)
        rest_lengths = np.array(rest_lengths)
        masses = np.ones(len(vertices))
        return (
            torch.tensor(vertices, dtype=torch.float32),
            torch.tensor(springs, dtype=torch.int32),
            torch.tensor(rest_lengths, dtype=torch.float32),
            torch.tensor(masses, dtype=torch.float32),
        )

    def get_spring_mass_visual(
        vertices,
        springs,
        rest_lengths,
        spring_forces,
        spring_isbreak,
        force_visual_scale=200,
    ):
        vertices = vertices.cpu().numpy()
        springs = springs.cpu().numpy()
        spring_forces = spring_forces.cpu().numpy()

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

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.paint_uniform_color([1, 0, 0])

        visuals = [lineset, pcd]
        return visuals

    table = o3d.io.read_point_cloud(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data/table.ply"
    )
    table.translate([0, 0, 0.1])
    init_vertices, init_springs, init_rest_lengths, init_masses = (
        get_spring_mass_from_pcd(table)
    )
    visuals = get_spring_mass_visual(
        init_vertices,
        init_springs,
        init_rest_lengths,
        spring_forces=torch.zeros(len(init_springs)),
        spring_isbreak=torch.zeros(len(init_springs)),
    )
    lineset, pcd = visuals

    mySystem = SpringMassSystem(
        init_vertices.cuda(),
        init_springs.cuda(),
        init_rest_lengths.cuda(),
        init_masses.cuda(),
        dt=5e-5,
        num_substeps=1000,
        spring_Y=3e6,
        dashpot_damping=100,
        drag_damping=1,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lineset)
    vis.add_geometry(pcd)
    ground_vertices = np.array([[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]])
    ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
    vis.add_geometry(ground_mesh)

    points_trajectories = []

    for _ in range(80):
        vertices, springs, rest_lengths, spring_forces = mySystem.step()
        points_trajectories.append(vertices)
        new_visuals = get_spring_mass_visual(
            vertices,
            springs,
            rest_lengths,
            torch.norm(spring_forces, dim=1),
            spring_isbreak=torch.zeros(len(init_springs)),
        )
        new_lineset, new_pcd = new_visuals
        lineset.points = new_lineset.points
        lineset.lines = new_lineset.lines
        lineset.colors = new_lineset.colors
        pcd.points = new_pcd.points
        vis.update_geometry(lineset)
        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
