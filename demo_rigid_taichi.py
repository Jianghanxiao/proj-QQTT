import open3d as o3d
import numpy as np
import taichi as ti


@ti.data_oriented
class RigidObjectSimulator:
    def __init__(
        self, center, r, init_masses, dt=5e-5, num_substeps=1000, drag_damping=1
    ):
        # Initialize the vertices
        # Here assume the mass are the same
        # Initialize the stuffs related to the translation
        self.n_vertices = len(r)
        self.masses = ti.field(dtype=ti.f32, shape=self.n_vertices)
        self.masses.from_numpy(init_masses)
        # Rigid body model
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.center.from_numpy(center.reshape(3, 1))
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.total_force = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Intialize the stuffs related to the rotation
        self.omega = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.total_torque = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Initialize the inverse mass and inverse inertia
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.r.from_numpy(r)
        self.total_mass = ti.field(dtype=ti.f32, shape=1)
        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)
        self._init_rigid_attributes()

        self.dt = dt
        self.num_substeps = num_substeps
        self.drag_damping = drag_damping
        import pdb

        pdb.set_trace()

    @ti.kernel
    def _init_rigid_attributes(self):
        print(self.n_vertices)
        for idx_vertice in range(self.n_vertices):
            self.total_mass[0] += self.masses[idx_vertice]

        for idx_vertice in range(self.n_vertices):
            self.inertia[0] += self.masses[idx_vertice] * (
                self.r[idx_vertice].norm_sqr() * ti.Matrix.identity(ti.f32, 3     )
                - self.r[idx_vertice].outer_product(self.r[idx_vertice])
            )

    def step(self):
        for i in range(self.num_substeps):
            self.substep()

        center = self.center.to_numpy()
        r = self.r.to_numpy()
        return center + r

    @ti.func
    def clear_forces(self):
        self.total_force[0] = ti.Vector([0.0, 0.0, 0.0])
        self.total_torque[0] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def substep(self):
        self.clear_forces()
        self.total_force[0] += ti.Vector([0.0, 0.0, -9.8]) * self.total_mass[0]


def demo1():
    # Test my rigid-object simulator
    # Load the table into taichi and create a simple spring-mass system
    table = o3d.io.read_point_cloud("data/table.ply")
    table.translate([0, 0, 0.1])

    init_vertices = np.asarray(table.points).astype(np.float32)
    center = init_vertices.mean(axis=0)
    init_masses = np.ones(len(init_vertices)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(init_vertices)

    mySystem = RigidObjectSimulator(
        center,
        init_vertices - center,
        init_masses,
        dt=5e-5,
        num_substeps=1000,
        drag_damping=1,
    )

    pass


if __name__ == "__main__":
    # ti.init(arch=ti.gpu)
    ti.init(arch=ti.cpu)
    # ti.init(arch=ti.cpu, cpu_max_num_threads=1)

    demo1()
