import torch
import torch.nn as nn
from qqtt.utils import logger, cfg
from .collision_detector import CollisionDetector


def nclamp(input, min, max):
    return input.clamp(min=min, max=max).detach() + input - input.detach()


# Differentialable Spring-Mass Simulator
class SpringMassSystem(nn.Module):
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        init_masks,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_velocities = None,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        super().__init__()
        self.device = cfg.device
        # Number of mass and springs
        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]
        # Initialization
        self.x = init_vertices
        if init_velocities is not None:
            self.v = init_velocities
        else:
            self.v = torch.zeros((self.n_vertices, 3), device=self.device)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
        self.masks = init_masks
        # Internal forces
        self.spring_forces = None
        # Use the log value to make it easier to learn
        self.spring_Y = nn.Parameter(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        # Parameters for the ground collision
        self.collide_elas = nn.Parameter(
            torch.tensor(
                collide_elas,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=True,
        )
        self.collide_fric = nn.Parameter(
            torch.tensor(
                collide_fric,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=True,
        )

        # Parameters for the object collision
        self.collide_object_elas = nn.Parameter(
            torch.tensor(
                collide_object_elas,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=True,
        )
        self.collide_object_fric = nn.Parameter(
            torch.tensor(
                collide_object_fric,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=True,
        )

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

        self.collision_dist = 0.05
        self.collisionDetector = CollisionDetector(len(self.x), self.collision_dist)
        self.object_collision_interval = 10
        self.object_interval_index = 0

        # representations for the bbx collisons, the object masks does not change
        unique_masks, inverse_indices = torch.unique(self.masks, return_inverse=True)
        self.num_objects = unique_masks.size(0)

        self.min_coords = torch.zeros(
            (self.num_objects, 3), dtype=self.x.dtype, device=self.device
        )
        self.max_coords = torch.zeros(
            (self.num_objects, 3), dtype=self.x.dtype, device=self.device
        )

        self.object_masks = []
        for i in range(self.num_objects):
            mask = inverse_indices == i
            self.object_masks.append(mask)
        self.object_masks = torch.stack(self.object_masks, dim=0)

        indices = torch.arange(self.num_objects)
        i, j = torch.meshgrid(indices, indices)
        pairs = torch.stack([i.flatten(), j.flatten()], dim=1)
        self.pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    def reset_system(self, init_vertices, init_springs, init_rest_lengths, init_masses, initial_velocities=None):
        logger.info(f"[SIMULATION]: Reset the Spring-Mass System")
        self.x = init_vertices
        if initial_velocities is not None:
            self.v = initial_velocities
        else:
            self.v = torch.zeros((self.n_vertices, 3), device=self.device)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
        self.object_collision_interval = 10
        self.object_interval_index = 0
        self.detach()

    def step(self):
        for i in range(self.num_substeps):
            self.substep()

        return (
            self.x,
            self.springs,
            self.rest_lengths,
            self.spring_forces,
        )

    def detach(self):
        # Detach all other stuff which is used across step
        self.x = self.x.detach().clone()
        self.v = self.v.detach().clone()
        if self.spring_forces is not None:
            self.spring_forces = self.spring_forces.detach().clone()
        torch.cuda.empty_cache()

    def substep(self):
        # One simulation step of the spring-mass system
        vertice_forces = torch.zeros((self.n_vertices, 3), device=self.device)

        # Add teh gravity force
        vertice_forces += self.masses[:, None] * torch.tensor(
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
            torch.exp(self.spring_Y)[:, None]
            * (torch.norm(dis, dim=1) / self.rest_lengths - 1)[:, None]
            * d
        )

        vertice_forces.index_add_(0, idx1, self.spring_forces)
        vertice_forces.index_add_(0, idx2, -self.spring_forces)

        # Apply the damping forces
        v_rel = torch.einsum("ij,ij->i", (self.v[idx2] - self.v[idx1]), d)
        dashpot_forces = self.dashpot_damping * v_rel[:, None] * d
        vertice_forces.index_add_(0, idx1, dashpot_forces)
        vertice_forces.index_add_(0, idx2, -dashpot_forces)

        # Update the velocity
        self.v += self.dt * vertice_forces / self.masses[:, None]
        self.v *= torch.exp(
            torch.tensor(-self.dt * self.drag_damping, device=self.device)
        )
        self.x += self.dt * self.v

        self.impulse_collision()
        # self.simple_ground_collision()

    def impulse_collision(self):
        # Check collisions with the ground
        self.ground_collision()
        if self.object_interval_index % self.object_collision_interval == 0:
            self.object_interval_index = 0
            if self.object_collision_interval == 1:
                flag = self.object_collision()
            else:
                if self.rough_box_collision():
                    flag = self.object_collision()
                else:
                    flag = False
            if flag:
                self.object_collision_interval = 1
            else:
                self.object_collision_interval = 10
        self.object_interval_index += 1

    def rough_box_collision(self):
        # Quick collision detection using bounding boxes
        for i in range(self.num_objects):
            self.min_coords[i] = self.x[self.object_masks[i]].min(dim=0)[0]
            self.max_coords[i] = self.x[self.object_masks[i]].max(dim=0)[0]

        overlap_x = (
            self.max_coords[self.pairs[:, 0], 0] + self.collision_dist
            >= self.min_coords[self.pairs[:, 1], 0]
        ) & (
            self.min_coords[self.pairs[:, 0], 0]
            <= self.max_coords[self.pairs[:, 1], 0] + self.collision_dist
        )
        overlap_y = (
            self.max_coords[self.pairs[:, 0], 1] + self.collision_dist
            >= self.min_coords[self.pairs[:, 1], 1]
        ) & (
            self.min_coords[self.pairs[:, 0], 1]
            <= self.max_coords[self.pairs[:, 1], 1] + self.collision_dist
        )
        overlap_z = (
            self.max_coords[self.pairs[:, 0], 2] + self.collision_dist
            >= self.min_coords[self.pairs[:, 1], 2]
        ) & (
            self.min_coords[self.pairs[:, 0], 2]
            <= self.max_coords[self.pairs[:, 1], 2] + self.collision_dist
        )

        overlap = overlap_x & overlap_y & overlap_z

        return overlap.any()

    def object_collision(self):
        collisions = self.collisionDetector.reset(self.x, self.masks)

        if len(collisions) == 0:
            return False

        idx1 = collisions[:, 0]
        idx2 = collisions[:, 1]
        x1 = self.x[idx1]
        x2 = self.x[idx2]
        v1 = self.v[idx1]
        v2 = self.v[idx2]
        m1 = self.masses[idx1]
        m2 = self.masses[idx2]
        dis = x2 - x1
        relative_velocity = v2 - v1

        collision_normal = dis / torch.norm(dis, dim=1, keepdim=True)
        # Only deal with collision when the velocity is going into the object
        mask = torch.einsum("ij,ij->i", relative_velocity, collision_normal) < 0
        if not mask.any():
            return False
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        x1 = x1[mask]
        x2 = x2[mask]
        v1 = v1[mask]
        v2 = v2[mask]
        m1 = m1[mask]
        m2 = m2[mask]
        collision_normal = collision_normal[mask]
        relative_velocity = relative_velocity[mask]

        v_rel_n = (relative_velocity * collision_normal).sum(
            dim=1, keepdim=True
        ) * collision_normal

        impulse_n = (
            -(1 + nclamp(self.collide_object_elas, min=0.0, max=1.0))
            * v_rel_n
            / (1 / m1 + 1 / m2)[:, None]
        )

        v_rel_t = relative_velocity - v_rel_n

        a = torch.max(
            torch.tensor(0.0, device=self.device),
            1
            - nclamp(self.collide_object_fric, min=0.0, max=1.0)
            * (1 + nclamp(self.collide_object_elas, min=0.0, max=1.0))
            * v_rel_n.norm(dim=1)
            / v_rel_t.norm(dim=1),
        )[:, None]

        impulse_t = (a - 1) * v_rel_t / (1 / m1 + 1 / m2)[:, None]

        J = impulse_n + impulse_t
        self.v[idx1] -= J / m1[:, None]
        self.v[idx2] += J / m2[:, None]

        return True

    def ground_collision(self):
        normal = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Check which vertices are below the ground and have velocity components going downward
        collision_mask = (self.x[:, 2] < 0) & (
            torch.einsum("ij,j->i", self.v, normal) < 0
        )

        if collision_mask.any():
            # Select the vertices that are colliding
            v_i = self.v[collision_mask]

            # Calculate normal and tangential components of the velocity
            v_normal = torch.einsum("ij,j->i", v_i, normal)[:, None] * normal
            v_tao = v_i - v_normal
            v_normal_new = -nclamp(self.collide_elas, min=0.0, max=1.0) * v_normal

            # Calculate the new tangential velocity component with friction
            a = torch.max(
                torch.tensor(0.0, device=self.device),
                1
                - nclamp(self.collide_fric, min=0.0, max=1.0)
                * (1 + nclamp(self.collide_elas, min=0.0, max=1.0))
                * v_normal.norm(dim=1)
                / v_tao.norm(dim=1),
            )[:, None]

            v_tao_new = a * v_tao
            v_i_new = v_normal_new + v_tao_new

            # Update velocities and positions of colliding vertices
            self.v[collision_mask] = v_i_new
            self.x[collision_mask, 2] = 0
