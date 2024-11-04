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
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        init_velocities=None,
        collision_dist=0.04,
        num_object_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
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
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points is None
            num_object_points = self.n_vertices
        else:
            assert num_object_points is not None
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.controller_points = controller_points
        # Handle the controller mask
        self.object_mask = torch.zeros(
            self.n_vertices, dtype=torch.int, device=self.device
        )
        self.object_mask[: self.num_object_points] = 1
        self.controller_mask = 1 - self.object_mask

        self.no_object_collision = False
        if init_masks is None:
            self.masks = torch.zeros(
                self.n_vertices, dtype=torch.int64, device=self.device
            )
            self.no_object_collision = True
        else:
            self.masks = init_masks
            if torch.unique(self.masks).size(0) == 1:
                self.no_object_collision = True
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
            requires_grad=cfg.collision_learn,
        )
        self.collide_fric = nn.Parameter(
            torch.tensor(
                collide_fric,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=cfg.collision_learn,
        )

        # Parameters for the object collision
        self.collide_object_elas = nn.Parameter(
            torch.tensor(
                collide_object_elas,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=cfg.collision_learn,
        )
        self.collide_object_fric = nn.Parameter(
            torch.tensor(
                collide_object_fric,
                dtype=torch.float32,
                device=self.device,
            ),
            requires_grad=cfg.collision_learn,
        )

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

        self.collision_dist = collision_dist
        self.collisionDetector = CollisionDetector(
            num_object_points, self.collision_dist
        )
        self.object_collision_interval = 10
        self.object_interval_index = 0

        self.reverse_factor = 1.0 if not reverse_z else -1.0

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

    def reset_system(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        initial_velocities=None,
    ):
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

    def set_controller(self, frame_idx):
        self.original_control_point = self.controller_points[frame_idx - 1]
        self.target_control_point = self.controller_points[frame_idx]

    def step(self):
        for i in range(self.num_substeps):
            if not self.controller_points is None:
                # Set the control points in each substep
                self.x[self.num_object_points :] = (
                    self.original_control_point
                    + (self.target_control_point - self.original_control_point)
                    * (i + 1)
                    / self.num_substeps
                )
                self.v[self.num_object_points :] = torch.zeros(
                    (self.controller_points.size(1), 3), device=self.device
                )
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
        vertice_forces += (
            self.masses[:, None]
            * torch.tensor([0.0, 0.0, -9.8], device=self.device)
            * self.reverse_factor
        )
        # Calculate the spring forces
        spring_mask = torch.exp(self.spring_Y) > self.spring_Y_min

        idx1 = self.springs[spring_mask][:, 0]
        idx2 = self.springs[spring_mask][:, 1]
        x1 = self.x[idx1]
        x2 = self.x[idx2]
        dis = x2 - x1
        d = dis / torch.max(
            torch.norm(dis, dim=1)[:, None], torch.tensor(1e-6, device=self.device)
        )
        self.spring_forces = (
            nclamp(
                torch.exp(self.spring_Y[spring_mask])[:, None],
                min=None,
                max=self.spring_Y_max,
            )
            * (torch.norm(dis, dim=1) / self.rest_lengths[spring_mask] - 1)[:, None]
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

        self.impulse_collision()
        if self.collision_mask.any():
            # Only update the object points
            final_mask = torch.logical_and(self.object_mask, self.collision_mask)
            self.x[final_mask] += (
                self.toi[:, None] * self.v_old[final_mask]
                + (self.dt - self.toi)[:, None] * self.v[final_mask]
            )

        # Only update the object points
        final_mask = torch.logical_and(self.object_mask, ~self.collision_mask)
        self.x[final_mask] += self.dt * self.v[final_mask]

    def impulse_collision(self):
        # Check collisions with the ground
        if not self.no_object_collision:
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
        self.ground_collision()

    def rough_box_collision(self):
        with torch.no_grad():
            # Quick collision detection using bounding boxes
            for i in range(self.num_objects):
                # Make sure the points are not the controller points
                final_mask = torch.logical_and(self.object_mask, self.object_masks[i])
                self.min_coords[i] = self.x[final_mask].min(dim=0)[0]
                self.max_coords[i] = self.x[final_mask].max(dim=0)[0]

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
        with torch.no_grad():
            collisions = self.collisionDetector.reset(
                self.x[self.object_mask].detach().clone(), self.masks
            )

        # # The most naive collision detection
        # with torch.no_grad():
        #     distances = torch.cdist(self.x, self.x)
        #     collisions = torch.nonzero(
        #         (distances < self.collision_dist) & (distances > 0.0), as_tuple=False
        #     )
        #     collisions = collisions[collisions[:, 0] < collisions[:, 1]]

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
        with torch.no_grad():
            mask = torch.einsum("ij,ij->i", relative_velocity, collision_normal) < -1e-4
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
            - nclamp(self.collide_object_fric, min=0.0, max=2.0)
            * (1 + nclamp(self.collide_object_elas, min=0.0, max=1.0))
            * v_rel_n.norm(dim=1)
            / torch.max(v_rel_t.norm(dim=1), torch.tensor(1e-6, device=self.device)),
        )[:, None]

        impulse_t = (a - 1) * v_rel_t / (1 / m1 + 1 / m2)[:, None]

        J = impulse_n + impulse_t

        index_counts = torch.zeros(
            self.v.size(0), dtype=torch.float32, device=self.device
        )
        index_counts.index_add_(
            0, idx1, torch.ones_like(idx1, dtype=torch.float, device=self.device)
        )
        index_counts.index_add_(
            0, idx2, torch.ones_like(idx2, dtype=torch.float, device=self.device)
        )

        current_v = torch.zeros(self.v.size(), dtype=torch.float32, device=self.device)
        current_v.index_add_(0, idx1, -J / m1[:, None])
        current_v.index_add_(0, idx2, J / m2[:, None])

        current_v /= torch.where(
            index_counts[:, None] == 0, torch.tensor(1.0), index_counts[:, None]
        )

        self.v += current_v

        return True

    def ground_collision(self):
        self.v_old = self.v.clone()

        normal = torch.tensor([0.0, 0.0, 1.0], device=self.device) * self.reverse_factor

        # Check which vertices are below the ground and have velocity components going downward
        with torch.no_grad():
            collision_mask = (
                ((self.x[:, 2] + self.v[:, 2] * self.dt) * self.reverse_factor) < 0
            ) & (torch.einsum("ij,j->i", self.v, normal) < -1e-4)
            # Just handle the ground collision for non-controller points
            self.collision_mask = torch.logical_and(collision_mask, self.object_mask)

        if collision_mask.any():
            # Select the vertices that are colliding
            v_i = self.v[collision_mask]

            # Calculate the time of impact
            self.toi = -self.x[collision_mask, 2] / v_i[:, 2]

            # Calculate normal and tangential components of the velocity
            v_normal = torch.einsum("ij,j->i", v_i, normal)[:, None] * normal
            v_tao = v_i - v_normal
            v_normal_new = -nclamp(self.collide_elas, min=0.0, max=1.0) * v_normal

            # Calculate the new tangential velocity component with friction
            a = torch.max(
                torch.tensor(0.0, device=self.device),
                1
                - nclamp(self.collide_fric, min=0.0, max=2.0)
                * (1 + nclamp(self.collide_elas, min=0.0, max=1.0))
                * v_normal.norm(dim=1)
                / torch.max(v_tao.norm(dim=1), torch.tensor(1e-6, device=self.device)),
            )[:, None]

            v_tao_new = a * v_tao
            v_i_new = v_normal_new + v_tao_new

            # Update velocities and positions of colliding vertices
            self.v[collision_mask] = v_i_new
