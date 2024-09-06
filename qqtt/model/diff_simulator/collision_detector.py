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
        return self.get_collisions(collision_len)

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


# S = ti.root.pointer(ti.i, 10).dynamic(ti.j, 1024, chunk_size=32)
# x = ti.field(int)
# S.place(x)

# @ti.kernel
# def add_data():
#     for i in range(10):
#         for j in range(i):
#             x[i].append(j)
#         print(x[i].length())  # will print i

#     for i in range(10):
#         x[i].deactivate()
#         print(x[i].length())  # will print 0

# add_data()
# import pdb
# pdb.set_trace()
seed = 1234
torch.manual_seed(seed)
import time

start = time.time()
collisionDetector = CollisionDetector(100000, 0.001, 0.07)
points = torch.rand(100000, 3).cuda()
import pdb
pdb.set_trace()
collisions = collisionDetector.reset(points)
print(f"FINAL_Length: {len(collisions)}")
print("Time: ", time.time() - start)
print(collisions)
