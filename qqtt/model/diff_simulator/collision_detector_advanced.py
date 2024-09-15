# Refer to the code in https://github.com/Denver-Pilphis/taichi_dem/blob/submit/Denver-Pilphis_MuGdxy/dem.py
# After testing, slower than my current implementation

import taichi as ti
import taichi.math as tm
import torch

ti.init(arch=ti.gpu)

# =====================================
# Type Definition
# =====================================
Real = ti.f64
Integer = ti.i32
# Byte = ti.i8
Vector2 = ti.types.vector(2, Real)
Vector3 = ti.types.vector(3, Real)
Vector4 = ti.types.vector(4, Real)
Vector3i = ti.types.vector(3, Integer)
Vector2i = ti.types.vector(2, Integer)
Matrix3x3 = ti.types.matrix(3, 3, Real)


def next_pow2(x):
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


# ======================================
# Broad Phase Collision Detection
# ======================================
@ti.data_oriented
class PrefixSumExecutor:
    def __init__(self):
        self.tree: ti.SNode = None
        self.temp: ti.StructField = None

    def _resize_temp(self, n):
        ti.sync()
        if self.tree != None:
            if self.temp.shape[0] >= n:
                return
            else:
                pass
            # self.tree.destroy()
        # ti.sync()
        # realloc
        print(f"resize_prefix_sum_temp:{n}")
        fb = ti.FieldsBuilder()
        self.temp = ti.field(Integer)
        fb.dense(ti.i, n).place(self.temp)
        self.tree = fb.finalize()

    @ti.kernel
    def serial(self, output: ti.template(), input: ti.template()):
        n = input.shape[0]
        output[0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, n):
            output[i] = output[i - 1] + input[i - 1]

    @ti.kernel
    def _down(
        self, d: Integer, n: Integer, offset: ti.template(), output: ti.template()
    ):
        for i in range(n):
            if i < d:
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                output[bi] += output[ai]

    @ti.kernel
    def _up(self, d: Integer, n: Integer, offset: ti.template(), output: ti.template()):
        for i in range(n):
            if i < d:
                ai = offset * (2 * i + 1) - 1
                bi = offset * (2 * i + 2) - 1
                tmp = output[ai]
                output[ai] = output[bi]
                output[bi] += tmp

    @ti.kernel
    def _copy(self, n: Integer, output: ti.template(), input: ti.template()):
        for i in range(n):
            output[i] = input[i]

    @ti.kernel
    def _copy_and_clear(
        self, n: Integer, npad: Integer, temp: ti.template(), input: ti.template()
    ):
        for i in range(n):
            temp[i] = input[i]
        for i in range(n, npad):
            temp[i] = 0

    def parallel_fast(self, output, input, cal_total=False):
        ti.static_assert(
            next_pow2(input.shape[0]) == input.shape[0],
            "parallel_fast requires input count = 2**p",
        )
        n: ti.i32 = input.shape[0]
        d = n >> 1
        self._copy(n, output, input)
        offset = 1
        while d > 0:
            self._down(d, n, offset, output)
            offset <<= 1
            d >>= 1

        output[n - 1] = 0
        d = 1
        while d < n:
            offset >>= 1
            self._up(d, n, offset, output)
            d <<= 1
        if cal_total:
            return output[n - 1] + input[n - 1]

    def parallel(self, output, input, cal_total=False):
        n: ti.i32 = input.shape[0]
        npad = next_pow2(n)
        self._resize_temp(npad)
        self._copy_and_clear(n, npad, self.temp, input)
        d = npad >> 1
        offset = 1
        while d > 0:
            self._down(d, npad, offset, self.temp)
            offset <<= 1
            d >>= 1

        self.temp[npad - 1] = 0
        d = 1
        while d < npad:
            offset >>= 1
            self._up(d, npad, offset, self.temp)
            d <<= 1
        self._copy(n, output, self.temp)
        if cal_total:
            return output[n - 1] + input[n - 1]


@ti.data_oriented
class BPCD:
    """
    Broad Phase Collision Detection
    """

    IGNORE_USER_DATA = -1
    ExplicitCollisionPair = 1
    Implicit = 0

    @ti.dataclass
    class HashCell:
        offset: Integer
        count: Integer
        current: Integer

    def __init__(
        self,
        particle_count: Integer,
        max_radius: Real,
        domain_min: Vector3,
        hash_table_size: Integer = 1 << 22,
        type=Implicit,
    ):
        self.type = type
        self.cell_size = max_radius * 4
        self.domain_min = domain_min
        self.hash_table = BPCD.HashCell.field(shape=hash_table_size)
        self.particle_id = ti.field(Integer, particle_count)

        self.pse = PrefixSumExecutor()

    def reset(self, domain_min):
        self.domain_min = domain_min

    def detect_collision(self, positions, collision_resolve_callback=None):
        """
        positions: field of Vector3
        bounding_sphere_radius: field of Real
        collision_resolve_callback: func(i:ti.i32, j:ti.i32, userdata) -> None
        """
        self._setup_collision(positions)

        self.pse.parallel_fast(self.hash_table.offset, self.hash_table.count)

        self._put_particles(positions)

        if self.type == BPCD.Implicit or collision_resolve_callback != None:
            self._solve_collision(positions, collision_resolve_callback)

    @ti.func
    def _count_particles(self, position: Vector3):
        ht = ti.static(self.hash_table)
        count = ti.atomic_add(ht[self.hash_codef(position)].count, 1)

    @ti.kernel
    def _put_particles(self, positions: ti.template()):
        ht = ti.static(self.hash_table)
        pid = ti.static(self.particle_id)
        for i in positions:
            hash_cell = self.hash_codef(positions[i])
            loc = ti.atomic_add(ht[hash_cell].current, 1)
            offset = ht[hash_cell].offset
            pid[offset + loc] = i

    @ti.func
    def _clear_hash_cell(self, i: Integer):
        ht = ti.static(self.hash_table)
        ht[i].offset = 0
        ht[i].current = 0
        ht[i].count = 0

    @ti.kernel
    def _setup_collision(self, positions: ti.template()):
        ht = ti.static(self.hash_table)
        for i in ht:
            self._clear_hash_cell(i)
        for i in positions:
            self._count_particles(positions[i])

    @ti.kernel
    def _solve_collision(
        self, positions: ti.template(), collision_resolve_callback: ti.template()
    ):
        ht = ti.static(self.hash_table)
        for i in positions:
            o = positions[i]
            ijk = self.cell(o)
            xyz = self.cell_center(ijk)
            Zero = Vector3i(0, 0, 0)
            dxyz = Zero

            for k in ti.static(range(3)):
                d = o[k] - xyz[k]
                if d > 0:
                    dxyz[k] = 1
                else:
                    dxyz[k] = -1

            cells = [
                ijk,
                ijk + Vector3i(dxyz[0], 0, 0),
                ijk + Vector3i(0, dxyz[1], 0),
                ijk + Vector3i(0, 0, dxyz[2]),
                ijk + Vector3i(0, dxyz[1], dxyz[2]),
                ijk + Vector3i(dxyz[0], 0, dxyz[2]),
                ijk + Vector3i(dxyz[0], dxyz[1], 0),
                ijk + dxyz,
            ]

            for k in ti.static(range(len(cells))):
                hash_cell = ht[self.hash_code(cells[k])]
                if hash_cell.count > 0:
                    for idx in range(
                        hash_cell.offset, hash_cell.offset + hash_cell.count
                    ):
                        pid = self.particle_id[idx]
                        if pid > i:
                            collision_resolve_callback(i, pid)

    # https://stackoverflow.com/questions/1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints
    @ti.func
    def morton3d32(x: Integer, y: Integer, z: Integer) -> Integer:
        answer = 0
        x &= 0x3FF
        x = (x | x << 16) & 0x30000FF
        x = (x | x << 8) & 0x300F00F
        x = (x | x << 4) & 0x30C30C3
        x = (x | x << 2) & 0x9249249
        y &= 0x3FF
        y = (y | y << 16) & 0x30000FF
        y = (y | y << 8) & 0x300F00F
        y = (y | y << 4) & 0x30C30C3
        y = (y | y << 2) & 0x9249249
        z &= 0x3FF
        z = (z | z << 16) & 0x30000FF
        z = (z | z << 8) & 0x300F00F
        z = (z | z << 4) & 0x30C30C3
        z = (z | z << 2) & 0x9249249
        answer |= x | y << 1 | z << 2
        return answer

    @ti.func
    def hash_codef(self, xyz: Vector3):
        return self.hash_code(self.cell(xyz))

    @ti.func
    def hash_code(self, ijk: Vector3i):
        return BPCD.morton3d32(ijk[0], ijk[1], ijk[2]) % self.hash_table.shape[0]

    @ti.func
    def cell(self, xyz: Vector3):
        ijk = ti.floor((xyz - self.domain_min) / self.cell_size, Integer)
        return ijk

    @ti.func
    def cell_center(self, ijk: Vector3i):
        ret = Vector3(0, 0, 0)
        for i in ti.static(range(3)):
            ret[i] = (ijk[i] + 0.5) * self.cell_size + self.domain_min[i]
        return ret


@ti.data_oriented
class CollisionDetector:
    def __init__(self, num_points, radius, domain_min=Vector3(-2, -2, -2)):
        self.num_points = num_points
        self.radius = radius

        self.points = ti.Vector.field(3, dtype=ti.f32, shape=num_points)
        self.points_mask = ti.field(ti.i32, shape=num_points)

        self.collisions_x = ti.field(ti.i32)
        self.collision_SNode_x = ti.root.dynamic(ti.i, num_points, chunk_size=32).place(
            self.collisions_x
        )
        self.collisions_y = ti.field(ti.i32)
        self.collision_SNode_y = ti.root.dynamic(ti.i, num_points, chunk_size=32).place(
            self.collisions_y
        )

        self.domain_min = domain_min
        # Initialize the BPCD
        self.bpcd = BPCD(num_points, radius, domain_min, type=BPCD.Implicit)

    def reset(self, points, points_mask=None):
        if points_mask is not None:
            self.points_mask.from_torch(points_mask)
        else:
            self.points_mask.fill(1)
        self.points.from_torch(points)

        lower_bound = points.min(0)[0]
        self.bpcd.reset(
            Vector3(lower_bound[0].item(), lower_bound[1].item(), lower_bound[2].item())
        )

        self.bpcd.detect_collision(self.points, self._collision_resolve)

        collision_len = self.get_collision_len()

        collisions = self.get_collisions(collision_len)
        return collisions

    @ti.kernel
    def get_collision_len(self) -> ti.i32:
        return self.collisions_x.length()

    @ti.func
    def _collision_resolve(self, i, j):
        if (self.points_mask[i] != self.points_mask[j]) and (
            (self.points[i] - self.points[j]).norm() < self.radius
        ):
            self.collisions_x.append(i)
            self.collisions_y.append(j)

    @ti.kernel
    def extract_collisions(self, buffer: ti.types.ndarray(ti.i32, ndim=2)):
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
    points_mask = torch.randint(0, 2, (num_points,), dtype=torch.int32).cuda()

    start = time.time()
    collisionDetector = CollisionDetector(num_points, 0.001)

    for i in range(100):
        # print("Time: ", time.time() - start)
        # start = time.time()
        # import pdb
        # pdb.set_trace()
        collisions = collisionDetector.reset(points, points_mask)
        # print(f"FINAL_Length: {len(collisions)}")
    print("Time: ", time.time() - start)
    # print(collisions)


# test1()
