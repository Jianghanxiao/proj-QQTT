import taichi as ti
import open3d as o3d
import numpy as np


def get_spring_mass_from_pcd(pcd, raidus=0.05, max_neighbours=11):
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
    return vertices, springs, rest_lengths, masses


def get_spring_mass_visual(vertices, springs, rest_legnths):
    # Color the springs with force information
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(springs)
    line_colors = [
        np.array([0.0, 1.0, 0.0])
        * np.abs(
            (
                1
                - np.linalg.norm(vertices[springs[i][0]] - vertices[springs[i][1]])
                / rest_legnths[i]
            )
        )
        for i in range(len(springs))
    ]
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
        num_substeps=10,
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
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)

        self.dt = 5e-5
        self.num_substeps = num_substeps
        self.spring_Y = 3e4
        self.dashpot_damping = 1e4
        self.drag_damping = 1

    def step(self):
        for i in range(self.num_substeps):
            self.substep()
        
        return self.x.to_numpy(), self.springs.to_numpy(), self.rest_lengths.to_numpy()

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
            idx1, idx2 = self.springs[idx_spring]
            x1, x2 = self.x[idx1], self.x[idx2]
            dis = x2 - x1
            d = dis.normalized()
            # Calculate the spring force
            force = (
                self.spring_Y
                * (dis.norm() / self.rest_lengths[idx_spring] - 1)
                * d
            )
            
            self.forces[idx1] += force
            self.forces[idx2] -= force
            # # Calculate the damping force
            # v_rel = (self.v[idx2] - self.v[idx1]).dot(d)
            # dashpot_force = self.dashpot_damping * v_rel * d
            # # self.forces[idx1] += dashpot_force
            # self.forces[idx2] -= dashpot_force

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
    table = o3d.io.read_point_cloud("data/table.ply")
    table.translate([0, 0, 0.3])
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    # o3d.visualization.draw_geometries([table, coordinate])
    init_vertices, init_springs, init_rest_lengths, init_masses = (
        get_spring_mass_from_pcd(table)
    )
    lineset, pcd = get_spring_mass_visual(init_vertices, init_springs, init_rest_lengths)
    # o3d.visualization.draw_geometries(visuals)

    mySystem = SpringMassSystem_taichi(
        init_vertices, init_springs, init_rest_lengths, init_masses
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(lineset)
    vis.add_geometry(pcd)
    


    for i in range(3000):
        vertices, springs, rest_lengths = mySystem.step()
        new_lineset, new_pcd = get_spring_mass_visual(vertices, springs, rest_lengths)
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
    # ti.init(arch=ti.gpu)
    ti.init(arch=ti.cpu, cpu_max_num_threads=1)

    demo1()
