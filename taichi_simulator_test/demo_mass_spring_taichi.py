import open3d as o3d
import numpy as np
import taichi as ti


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
        vertices.astype(np.float32),
        springs.astype(np.int32),
        rest_lengths.astype(np.float32),
        masses.astype(np.float32),
    )


def get_spring_mass_visual(
    vertices, springs, rest_lengths, spring_forces, spring_isbreak, force_visual_scale=200
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
            line_colors.append(np.array([0.0, 1.0, 0.0]) * spring_forces[i] / force_visual_scale)
    lineset.lines = o3d.utility.Vector2iVector(temp_springs)
    lineset.colors = o3d.utility.Vector3dVector(line_colors)

    # Color the vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.paint_uniform_color([1, 0, 0])

    visuals = [lineset, pcd]
    return visuals


@ti.data_oriented
class SpringMassSystem_taichi:
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
        break_force_limit=200000,
    ):
        # Initialize the vertices and springs
        # Here assume the mass are the same
        self.n_vertices = len(init_vertices)
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.x.from_numpy(init_vertices)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.n_springs = len(init_springs)
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=self.n_springs)
        self.springs.from_numpy(init_springs)
        self.rest_lengths = ti.field(dtype=ti.f32, shape=self.n_springs)
        self.rest_lengths.from_numpy(init_rest_lengths)
        self.masses = ti.field(dtype=ti.f32, shape=self.n_vertices)
        self.masses.from_numpy(init_masses)
        # spring_forces is used as a middle stage to store the spring forces
        self.spring_isbreak = ti.field(dtype=ti.i32, shape=self.n_springs)
        self.spring_forces = ti.field(dtype=ti.f32, shape=self.n_springs)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)

        self.dt = dt
        self.num_substeps = num_substeps
        self.spring_Y = spring_Y
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.break_force_limit = break_force_limit

    def step(self):
        for i in range(self.num_substeps):
            self.substep()

        return (
            self.x.to_numpy(),
            self.springs.to_numpy(),
            self.rest_lengths.to_numpy(),
            self.spring_forces.to_numpy(),
            self.spring_isbreak.to_numpy(),
        )

    @ti.func
    def clear_forces(self):
        for i in self.forces:
            self.forces[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def substep(self):
        self.clear_forces()
        for idx_vertice in range(self.n_vertices):
            self.forces[idx_vertice] += (
                ti.Vector([0.0, 0.0, -9.8]) * self.masses[idx_vertice]
            )

        for idx_spring in range(self.n_springs):
            if self.spring_isbreak[idx_spring] == 1:
                continue
            idx1, idx2 = self.springs[idx_spring]
            x1, x2 = self.x[idx1], self.x[idx2]
            dis = x2 - x1
            d = dis.normalized()
            # Calculate the spring force
            force = self.spring_Y * (dis.norm() / self.rest_lengths[idx_spring] - 1) * d

            self.forces[idx1] += force
            self.forces[idx2] -= force
            # Calculate the damping force
            v_rel = (self.v[idx2] - self.v[idx1]).dot(d)
            dashpot_force = self.dashpot_damping * v_rel * d
            self.forces[idx1] += dashpot_force
            self.forces[idx2] -= dashpot_force

            # Check if the spring is broken
            self.spring_forces[idx_spring] = force.norm()
            if self.spring_forces[idx_spring] > self.break_force_limit:
                self.spring_isbreak[idx_spring] = 1

        for idx_vertice in range(self.n_vertices):
            self.v[idx_vertice] += (
                self.forces[idx_vertice] / self.masses[idx_vertice] * self.dt
            )
            self.v[idx_vertice] *= ti.exp(-self.dt * self.drag_damping)
            self.x[idx_vertice] += self.v[idx_vertice] * self.dt

            # Assuming only ground for now
            if self.x[idx_vertice][2] < 0:
                self.x[idx_vertice][2] = 0
                self.v[idx_vertice][2] = 0


def demo1():
    # Load the table into taichi and create a simple spring-mass system
    table = o3d.io.read_point_cloud("taichi_simulator_test/data/table.ply")
    table.translate([0, 0, 0.1])
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([table, coordinate])
    init_vertices, init_springs, init_rest_lengths, init_masses = (
        get_spring_mass_from_pcd(table)
    )
    lineset, pcd = get_spring_mass_visual(
        init_vertices,
        init_springs,
        init_rest_lengths,
        spring_forces=np.zeros(len(init_springs)),
        spring_isbreak=np.zeros(len(init_springs)),
    )
    # o3d.visualization.draw_geometries(visuals)

    # For spring setting
    mySystem = SpringMassSystem_taichi(
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt=5e-5,
        num_substeps=1000,
        spring_Y=3e4,
        dashpot_damping=100,
        drag_damping=1,
    )
    # # For fake rigid setting
    # mySystem = SpringMassSystem_taichi(
    #     init_vertices,
    #     init_springs,
    #     init_rest_lengths,
    #     init_masses,
    #     dt=5e-6,
    #     num_substeps=1000,
    #     spring_Y=3e6,
    #     dashpot_damping=100,
    #     drag_damping=10,
    # )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lineset)
    vis.add_geometry(pcd)
    # Define ground plane vertices
    ground_vertices = np.array([[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]])

    # Define ground plane triangular faces
    ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])

    # Create Open3D mesh object
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
    vis.add_geometry(ground_mesh)

    points_trajectories = []

    for i in range(80):
        vertices, springs, rest_lengths, spring_forces, spring_isbreak = mySystem.step()
        points_trajectories.append(vertices)
        new_lineset, new_pcd = get_spring_mass_visual(
            vertices, springs, rest_lengths, spring_forces, spring_isbreak
        )
        lineset.points = new_lineset.points
        lineset.lines = new_lineset.lines
        lineset.colors = new_lineset.colors
        pcd.points = new_pcd.points
        vis.update_geometry(lineset)
        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        # import pdb
        # pdb.set_trace()
    vis.destroy_window()

    # Save points_trajectories to a npy file
    points_trajectories = np.array(points_trajectories)
    points_trajectories = np.transpose(points_trajectories, (1, 0, 2))
    np.save("points_trajectories_spring.npy", points_trajectories)

def demo2():
    # Test the breaking phenomenon
    # Load the table into taichi and create a simple spring-mass system
    table = o3d.io.read_point_cloud("data/table.ply")
    table.translate([0, 0, 0.1])
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([table, coordinate])
    init_vertices, init_springs, init_rest_lengths, init_masses = (
        get_spring_mass_from_pcd(table)
    )
    lineset, pcd = get_spring_mass_visual(
        init_vertices,
        init_springs,
        init_rest_lengths,
        spring_forces=np.zeros(len(init_springs)),
        spring_isbreak=np.zeros(len(init_springs)),
    )
    # o3d.visualization.draw_geometries(visuals)

    # For spring setting
    mySystem = SpringMassSystem_taichi(
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt=5e-5,
        num_substeps=1000,
        spring_Y=3e4,
        dashpot_damping=100,
        drag_damping=1,
        break_force_limit=1500,
    )

    # # For fake rigid setting
    # mySystem = SpringMassSystem_taichi(
    #     init_vertices,
    #     init_springs,
    #     init_rest_lengths,
    #     init_masses,
    #     dt=5e-6,
    #     num_substeps=1000,
    #     spring_Y=3e6,
    #     dashpot_damping=100,
    #     drag_damping=10,
    #     break_force_limit=1500,
    # )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lineset)
    vis.add_geometry(pcd)
    # Define ground plane vertices
    ground_vertices = np.array([[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0]])

    # Define ground plane triangular faces
    ground_triangles = np.array([[0, 2, 1], [0, 3, 2]])

    # Create Open3D mesh object
    ground_mesh = o3d.geometry.TriangleMesh()
    ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground_mesh.paint_uniform_color([1, 211 / 255, 139 / 255])
    vis.add_geometry(ground_mesh)

    for i in range(3000):
        vertices, springs, rest_lengths, spring_forces, spring_isbreak = mySystem.step()
        # print(spring_forces.min(), spring_forces.max(), spring_forces.mean(), np.median(spring_forces))
        new_lineset, new_pcd = get_spring_mass_visual(
            vertices, springs, rest_lengths, spring_forces, spring_isbreak
        )
        lineset.points = new_lineset.points
        lineset.lines = new_lineset.lines
        lineset.colors = new_lineset.colors
        pcd.points = new_pcd.points
        vis.update_geometry(lineset)
        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        # import pdb
        # pdb.set_trace()
    vis.destroy_window()


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    # ti.init(arch=ti.cpu, cpu_max_num_threads=1)

    demo1()
    # demo2()
