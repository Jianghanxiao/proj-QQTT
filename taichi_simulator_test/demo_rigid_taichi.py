# The test script for the rigid object simulator implemented in Taichi

import open3d as o3d
import numpy as np
import taichi as ti
import math


@ti.func
def SetToRotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return ti.Matrix(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy + qz * qw),
                2 * (qx * qz - qy * qw),
            ],
            [
                2 * (qx * qy - qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz + qx * qw),
            ],
            [
                2 * (qx * qz + qy * qw),
                2 * (qy * qz - qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    ).transpose()


@ti.data_oriented
class RigidObjectSimulator:
    def __init__(
        self,
        center,
        r,
        init_masses,
        dt=5e-5,
        num_substeps=100,
        drag_damping=1,
        collide_elas=0.5,
        collide_fric=0.3,
    ):
        # Initialize the vertices
        # Here assume the mass are the same
        # Initialize the stuffs related to the translation
        self.n_vertices = len(r)
        self.masses = ti.field(dtype=ti.f32, shape=self.n_vertices)
        self.masses.from_numpy(init_masses)
        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        # Rigid body model
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.center.from_numpy(center.reshape(3, 1))
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.total_force = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Intialize the stuffs related to the rotation
        self.omega = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self.rot_q_inc = ti.Vector.field(4, dtype=ti.f32, shape=1)
        self.rot_inc = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)
        self.total_torque = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Initialize the inverse mass and inverse inertia
        # r and interia here is updated every step
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.r.from_numpy(r)
        self.total_mass = ti.field(dtype=ti.f32, shape=1)
        self.inertia = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)
        self._init_rigid_attributes()

        self.dt = dt
        self.num_substeps = num_substeps
        self.drag_damping = drag_damping
        # Parameters for collision
        self.collide_elas = collide_elas
        self.collide_fric = collide_fric

    @ti.kernel
    def _init_rigid_attributes(self):
        print(self.n_vertices)
        for idx_vertice in range(self.n_vertices):
            self.total_mass[0] += self.masses[idx_vertice]

        for idx_vertice in range(self.n_vertices):
            self.inertia[0] += self.masses[idx_vertice] * (
                self.r[idx_vertice].norm_sqr() * ti.Matrix.identity(ti.f32, 3)
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
        # Initialize the forces and torques
        for idx_vertice in range(self.n_vertices):
            self.forces[idx_vertice] = ti.Vector([0.0, 0.0, 0.0])
        self.total_force[0] = ti.Vector([0.0, 0.0, 0.0])
        self.total_torque[0] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def substep(self):
        self.clear_forces()
        # Process the collision
        # self.process_collision_avg()
        self.process_collision()
        # Process the translation
        for idx_vertice in range(self.n_vertices):
            self.forces[idx_vertice] += (
                ti.Vector([0.0, 0.0, -9.8]) * self.masses[idx_vertice]
            )
            self.total_force[0] += self.forces[idx_vertice]
        self.v[0] += self.total_force[0] * self.dt / self.total_mass[0]
        self.center[0] += self.v[0] * self.dt
        # Process the rotation
        for idx_vertice in range(self.n_vertices):
            # Here is a option to add torque to make the case where center is not gradity center work; However, it will make the system complicated
            # The normal things work as long as the center is the gravity center
            # self.total_torque[0] += self.r[idx_vertice].cross(
            #     self.forces[idx_vertice]
            # ) - self.masses[idx_vertice] / self.total_mass[0] * self.r[
            #     idx_vertice
            # ].cross(
            #     self.total_force[0]
            # )
            self.total_torque[0] += self.r[idx_vertice].cross(self.forces[idx_vertice])
        self.omega[0] += self.dt * self.inertia[0].inverse() @ self.total_torque[0]
        # Based on current omega, update the Interia and the r
        # This is the same to the forumula treating current q is [0, 0, 0, 1] (xyzw)
        self.rot_q_inc[0][3] = 1
        self.rot_q_inc[0][:3] = self.dt / 2 * self.omega[0]
        self.rot_q_inc[0] = self.rot_q_inc[0].normalized()
        # Equal to below code to get the quaternion using axis angle
        # radian = (self.dt * self.omega[0]).norm()
        # direction = (self.dt * self.omega[0]).normalized()
        # self.rot_q_inc[0][3] = ti.cos(radian / 2)
        # self.rot_q_inc[0][0] = direction[0] * ti.sin(radian / 2)
        # self.rot_q_inc[0][1] = direction[1] * ti.sin(radian / 2)
        # self.rot_q_inc[0][2] = direction[2] * ti.sin(radian / 2)
        self.rot_inc[0] = SetToRotate(self.rot_q_inc[0])

        for i in range(self.n_vertices):
            self.r[i] = self.rot_inc[0] @ self.r[i]
        self.inertia[0] = (
            self.rot_inc[0] @ self.inertia[0] @ self.rot_inc[0].transpose()
        )

        # Apply the drag damping
        decreasing_ratio = math.exp(-self.dt * self.drag_damping)
        self.v[0] *= decreasing_ratio
        self.omega[0] *= decreasing_ratio

    @ti.func
    def process_collision_avg(self):
        # An alternative way to process the collision using the mean position of the collision points
        avg_r = ti.Vector([0.0, 0.0, 0.0])
        num_vertices = 0
        normal = ti.Vector([0.0, 0.0, 1.0])
        for idx_vertice in range(self.n_vertices):
            vertice = self.center[0] + self.r[idx_vertice]
            v_i = self.v[0] + self.omega[0].cross(self.r[idx_vertice])
            if vertice[2] < 0 and v_i.dot(normal) < 0:
                avg_r += self.r[idx_vertice]
                num_vertices += 1

        if num_vertices > 0:
            avg_r /= num_vertices

            v_i = self.v[0] + self.omega[0].cross(avg_r)

            v_normal = v_i.dot(normal) * normal
            v_tao = v_i - v_normal

            v_normal_new = -self.collide_elas * v_normal
            a = ti.max(
                0.0,
                1
                - self.collide_fric
                * (1 + self.collide_elas)
                * v_normal.norm()
                / v_tao.norm(),
            )
            v_tao_new = a * v_tao
            v_i_new = v_normal_new + v_tao_new

            avg_r_matrix = ti.Matrix(
                [
                    [0.0, -avg_r.z, avg_r.y],
                    [avg_r.z, 0.0, -avg_r.x],
                    [-avg_r.y, avg_r.x, 0.0],
                ]
            )

            K = (
                1 / self.total_mass[0] * ti.Matrix.identity(ti.f32, 3)
                - avg_r_matrix @ self.inertia[0].inverse() @ avg_r_matrix
            )

            j = K.inverse() @ (v_i_new - v_i)

            # print(num_vertices)
            # print(j / self.total_mass[0], self.inertia[0].inverse() @ avg_r_matrix @ j)

            self.v[0] += j / self.total_mass[0]
            self.omega[0] += self.inertia[0].inverse() @ avg_r_matrix @ j

    @ti.func
    def process_collision(self):
        normal = ti.Vector([0.0, 0.0, 1.0])
        v_inc = ti.Vector([0.0, 0.0, 0.0])
        omega_inc = ti.Vector([0.0, 0.0, 0.0])
        num_vertices = 0
        for idx_vertice in range(self.n_vertices):
            vertice = self.center[0] + self.r[idx_vertice]
            r_i = self.r[idx_vertice]
            v_i = self.v[0] + self.omega[0].cross(r_i)
            if vertice[2] < 0 and v_i.dot(normal) < 0:
                num_vertices += 1
                v_normal = v_i.dot(normal) * normal
                v_tao = v_i - v_normal

                v_normal_new = -self.collide_elas * v_normal
                a = ti.max(
                    0.0,
                    1
                    - self.collide_fric
                    * (1 + self.collide_elas)
                    * v_normal.norm()
                    / v_tao.norm(),
                )
                v_tao_new = a * v_tao
                v_i_new = v_normal_new + v_tao_new

                r_matrix = ti.Matrix(
                    [
                        [0.0, -r_i.z, r_i.y],
                        [r_i.z, 0.0, -r_i.x],
                        [-r_i.y, r_i.x, 0.0],
                    ]
                )

                K = (
                    1 / self.total_mass[0] * ti.Matrix.identity(ti.f32, 3)
                    - r_matrix @ self.inertia[0].inverse() @ r_matrix
                )

                j = K.inverse() @ (v_i_new - v_i)

                v_inc += j / self.total_mass[0]
                omega_inc += self.inertia[0].inverse() @ r_matrix @ j
        if num_vertices > 0:
            # print(num_vertices)
            # print(v_inc, omega_inc)
            self.v[0] += v_inc / num_vertices
            self.omega[0] += omega_inc / num_vertices


def demo1():
    # Test my rigid-object simulator
    # Load the table into taichi and create a simple spring-mass system
    table = o3d.io.read_point_cloud(
        "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/taichi_simulator_test/data/table.ply"
    )
    # R = table.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    # table.rotate(R, center=(0, 0, 0))
    table.translate([0, 0, 2])

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
        drag_damping=0,
    )

    # Construct the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1, origin=[0, 0, 0]
    # )
    # vis.add_geometry(coordinate)
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

    view_control = vis.get_view_control()
    view_control.set_front([-1, 0, 0.5])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(3)

    points_trajectories = []

    for i in range(50):
        points = mySystem.step()
        points_trajectories.append(points)
        # print(spring_forces.min(), spring_forces.max(), spring_forces.mean(), np.median(spring_forces))
        pcd.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        # import pdb
        # pdb.set_trace()
    vis.destroy_window()

    # Save points_trajectories to a npy file
    points_trajectories = np.array(points_trajectories)
    points_trajectories = np.transpose(points_trajectories, (1, 0, 2))
    # np.save("taichi_simulator_test/points_trajectories_rigid.npy", points_trajectories)


if __name__ == "__main__":
    # ti.init(arch=ti.gpu)
    # ti.init(arch=ti.cpu)
    ti.init(arch=ti.cpu, cpu_max_num_threads=1)

    demo1()
