import taichi as ti
import torch
import numpy as np
import time

ti.init(arch=ti.gpu)


@ti.data_oriented
class CollisionDetector:
    def __init__(self, num_points, radius, grid_count=60):
        self.num_points = num_points
        self.radius = radius
        self.grid_count = grid_count

        self.points = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
        self.points_mask = ti.field(ti.i32, shape=num_points)

        self.max_grid_size = 500
        self.grid = ti.field(ti.i32)
        self.grid_SNodes = ti.root.pointer(
            ti.ijk, (self.grid_count, self.grid_count, self.grid_count)
        )
        self.block = self.grid_SNodes.dense(ti.l, self.max_grid_size).place(self.grid)
        self.grid_point_count = ti.field(
            ti.i32, shape=(self.grid_count, self.grid_count, self.grid_count)
        )
        self.grid_point_count.fill(0)

        self.max_collisions = num_points * 10
        self.collisions = ti.Vector.field(2, dtype=ti.i32, shape=self.max_collisions)
        self.collisions_count = ti.field(dtype=ti.i32, shape=())
        self.collisions_count[None] = 0

    def reset(self, points, points_mask=None):
        if points_mask is not None:
            self.points_mask.from_torch(points_mask)
        else:
            self.points_mask.fill(1)
        self.points.from_torch(points)

        # Have some extra grids to avoid boundary issues
        grid_size = (points.max(0)[0] - points.min(0)[0]) / (self.grid_count - 10)
        self.grid_size = ti.Vector(
            [
                max(self.radius, grid_size[0].item()),
                max(self.radius, grid_size[1].item()),
                max(self.radius, grid_size[2].item()),
            ]
        )
        self.lower_bound = ti.Vector(
            [
                points.min(0)[0][0].item(),
                points.min(0)[0][1].item(),
                points.min(0)[0][2].item(),
            ]
        )
        ti.deactivate_all_snodes()
        self.grid_point_count.fill(0)
        self.collisions_count[None] = 0
        self.assign_points_to_grid()
        self.detect_collisions()
        collisions = self.collisions.to_torch(device="cuda")[
            : self.collisions_count[None]
        ]
        return collisions

    @ti.kernel
    def assign_points_to_grid(self):
        for i in range(self.num_points):
            cell = ((self.points[i] - self.lower_bound) / self.grid_size).cast(ti.i32)
            cell = ti.max(ti.min(cell, self.grid_count - 1), 0)
            index = ti.atomic_add(self.grid_point_count[cell[0], cell[1], cell[2]], 1)
            if index >= self.max_grid_size:
                print(f"Warning: Too many points ({index}) in a cell !!!!!!!!!!!!")
            self.grid[cell[0], cell[1], cell[2], index] = i

    @ti.func
    def check_collision(self, p1, p2):
        return (self.points_mask[p1] != self.points_mask[p2]) and (
            (self.points[p1] - self.points[p2]).norm() < self.radius
        )

    @ti.kernel
    def detect_collisions(self):
        for i, j, k in self.grid_SNodes:
            length = self.grid_point_count[i, j, k]
            if length > 0:
                # Check collisions within the same cell
                for ii in range(length):
                    p1 = self.grid[i, j, k, ii]
                    for jj in range(ii + 1, length):
                        p2 = self.grid[i, j, k, jj]
                        if self.check_collision(p1, p2):
                            index = ti.atomic_add(self.collisions_count[None], 1)
                            if index >= self.max_collisions:
                                print(f"Warning: Too many collisions ({index}) !!!!!!!!!!!!")
                            if p1 < p2:
                                self.collisions[index][0] = p1
                                self.collisions[index][1] = p2
                            else:
                                self.collisions[index][0] = p2
                                self.collisions[index][1] = p1
                # Check collisions with neighboring cells
                for di, dj, dk in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (
                        0 <= ni < self.grid_count
                        and 0 <= nj < self.grid_count
                        and 0 <= nk < self.grid_count
                        and self.grid_point_count[ni, nj, nk] > 0
                    ):
                        neighbor_length = self.grid_point_count[ni, nj, nk]
                        for ii in range(length):
                            p1 = self.grid[i, j, k, ii]
                            for jj in range(neighbor_length):
                                p2 = self.grid[ni, nj, nk, jj]
                                if p1 < p2 and self.check_collision(p1, p2):
                                    index = ti.atomic_add(
                                        self.collisions_count[None], 1
                                    )
                                    if index >= self.max_collisions:
                                        print(
                                            "Warning: Too many collisions !!!!!!!!!!!!"
                                        )
                                    self.collisions[index][0] = p1
                                    self.collisions[index][1] = p2


def test1():
    seed = 1234
    torch.manual_seed(seed)
    import time

    num_points = 100000
    points = torch.rand(num_points, 3).cuda()
    points_mask = torch.randint(0, 100, (num_points,), dtype=torch.int32).cuda()
    start = time.time()
    # collisionDetector = CollisionDetector(num_points, 0.001, 0.07)
    collisionDetector = CollisionDetector(num_points, 0.001)
    for i in range(1):
        # print("Time: ", time.time() - start)
        # start = time.time()
        # import pdb
        # pdb.set_trace()
        collisions = collisionDetector.reset(points, points_mask)
        # print(f"FINAL_Length: {len(collisions)}")
    print("Time: ", time.time() - start)
    print(len(collisions))
    # # Use open3d to visualize the points and the collisions
    # import open3d as o3d

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    # colors = np.zeros((num_points, 3))
    # colors[collisions.cpu().numpy()] = [0.2, 0.2, 0.2]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # # Draw sphere for the collision points
    # spheres = []
    # for i in range(len(collisions)):
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    #     sphere.paint_uniform_color([1, 0, 0])
    #     sphere.translate(points[collisions[i][0]].cpu().numpy())
    #     spheres.append(sphere)
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    #     sphere.paint_uniform_color([1, 0, 0])
    #     sphere.translate(points[collisions[i][1]].cpu().numpy())
    #     spheres.append(sphere)

    # # collision_pcd = o3d.geometry.PointCloud()
    # # collision_pcd.points = o3d.utility.Vector3dVector(collision_points)
    # # collision_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(collision_points)))
    # o3d.visualization.draw_geometries([pcd] + spheres)


# test1()
