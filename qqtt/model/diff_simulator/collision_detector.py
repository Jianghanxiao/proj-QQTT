import taichi as ti
import torch
import numpy as np

ti.init(arch=ti.gpu)


@ti.data_oriented
class CollisionDetector:
    def __init__(self, num_points, radius, grid_size):
        self.num_points = num_points
        self.radius = radius
        self.grid_size = grid_size
        self.grid_count = (int(1 // grid_size) + 1) * 3

        self.points = ti.Vector.field(3, dtype=ti.f32, shape=num_points)

        self.grid = ti.field(ti.i32)
        self.grid_SNodes = ti.root.pointer(
            ti.ijk, (self.grid_count, self.grid_count, self.grid_count)
        )
        self.block = self.grid_SNodes.dynamic(
            ti.l, self.num_points, chunk_size=32
        ).place(self.grid)

        self.collisions_x = ti.field(ti.i32)
        self.collision_SNode_x = ti.root.dynamic(ti.i, num_points, chunk_size=32).place(
            self.collisions_x
        )
        self.collisions_y = ti.field(ti.i32)
        self.collision_SNode_y = ti.root.dynamic(ti.i, num_points, chunk_size=32).place(
            self.collisions_y
        )

    def reset(self, points):
        self.points.from_torch(points)
        ti.deactivate_all_snodes()
        self.assign_points_to_grid()
        collision_len = self.detect_collisions()
        collisions = self.get_collisions(collision_len)
        return collisions

    @ti.kernel
    def assign_points_to_grid(self):
        for i in range(self.num_points):
            cell = (self.points[i] / self.grid_size).cast(ti.i32)
            cell = ti.max(ti.min(cell, self.grid_count - 1), 0)
            self.grid[cell[0], cell[1], cell[2]].append(i)

    @ti.func
    def check_collision(self, p1, p2):
        # print((self.points[p1] - self.points[p2]).norm())
        return (self.points[p1] - self.points[p2]).norm() < self.radius

    @ti.kernel
    def detect_collisions(self) -> ti.i32:
        for i, j, k in self.grid_SNodes:
            length = self.grid[i, j, k].length()
            if length > 0:
                # Check collisions within the same cell
                for ii in range(length):
                    p1 = self.grid[i, j, k, ii]
                    for jj in range(ii + 1, length):
                        p2 = self.grid[i, j, k, jj]
                        if self.check_collision(p1, p2):
                            if p1 < p2:
                                self.collisions_x.append(p1)
                                self.collisions_y.append(p2)
                            else:
                                self.collisions_x.append(p2)
                                self.collisions_y.append(p1)

                # Check collisions with neighboring cells
                for di, dj, dk in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (
                        0 <= ni < self.grid_count
                        and 0 <= nj < self.grid_count
                        and 0 <= nk < self.grid_count
                        and self.grid[ni, nj, nk].length() > 0
                    ):
                        neighbor_length = self.grid[ni, nj, nk].length()
                        for ii in range(length):
                            p1 = self.grid[i, j, k, ii]
                            for jj in range(neighbor_length):
                                p2 = self.grid[ni, nj, nk, jj]
                                if self.check_collision(p1, p2):
                                    if p1 < p2:
                                        self.collisions_x.append(p1)
                                        self.collisions_y.append(p2)
                                    else:
                                        self.collisions_x.append(p2)
                                        self.collisions_y.append(p1)
        return self.collisions_x.length()

    @ti.kernel
    def extract_collisions(self, buffer: ti.types.ndarray(ti.i32, ndim=2)):
        print(self.collisions_x.length())
        for i in range(self.collisions_x.length()):
            buffer[i, 0] = self.collisions_x[i]
            buffer[i, 1] = self.collisions_y[i]

    def get_collisions(self, collision_len):
        collisions_torch = torch.zeros(
            (collision_len, 2), dtype=torch.int32, device="cuda"
        )
        self.extract_collisions(collisions_torch)
        return torch.unique(collisions_torch, dim=0)


def test1():
    seed = 1234
    torch.manual_seed(seed)
    import time

    num_points = 100000
    points = torch.rand(num_points, 3).cuda()
    start = time.time()
    collisionDetector = CollisionDetector(num_points, 0.001, 0.07)
    for i in range(1):
        # print("Time: ", time.time() - start)
        # start = time.time()
        # import pdb
        # pdb.set_trace()
        collisions = collisionDetector.reset(points)
        # print(f"FINAL_Length: {len(collisions)}")
    print("Time: ", time.time() - start)
    print(collisions)
    # Use open3d to visualize the points and the collisions
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    colors = np.zeros((num_points, 3))
    colors[collisions.cpu().numpy()] = [0.2, 0.2, 0.2]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Draw sphere for the collision points
    spheres = []
    for i in range(len(collisions)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(points[collisions[i][0]].cpu().numpy())
        spheres.append(sphere)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.translate(points[collisions[i][1]].cpu().numpy())
        spheres.append(sphere)

    # collision_pcd = o3d.geometry.PointCloud()
    # collision_pcd.points = o3d.utility.Vector3dVector(collision_points)
    # collision_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(collision_points)))
    o3d.visualization.draw_geometries([pcd] + spheres)

test1()