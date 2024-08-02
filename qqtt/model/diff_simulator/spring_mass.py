import torch
import torch.nn as nn
from qqtt.utils import logger, cfg


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
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        super().__init__()
        self.device = cfg.device
        # Number of mass and springs
        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]
        # Initialization
        self.x = init_vertices
        self.v = torch.zeros((self.n_vertices, 3), device=self.device)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
        # Internal forces
        self.spring_forces = None
        # Use the log value to make it easier to learn
        self.spring_Y = nn.Parameter(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        # Parameters for the collision
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

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

    def reset_system(self, init_vertices, init_springs, init_rest_lengths, init_masses):
        logger.info(f"[SIMULATION]: Reset the Spring-Mass System")
        self.x = init_vertices
        self.v = torch.zeros((self.n_vertices, 3), device=self.device)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
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
            v_normal_new = -self.collide_elas * v_normal

            # Calculate the new tangential velocity component with friction
            a = torch.max(
                torch.tensor(0.0, device=self.device),
                1
                - self.collide_fric
                * (1 + self.collide_elas)
                * v_normal.norm(dim=1)
                / v_tao.norm(dim=1),
            )[:, None]

            v_tao_new = a * v_tao
            v_i_new = v_normal_new + v_tao_new

            # Update velocities and positions of colliding vertices
            self.v[collision_mask] = v_i_new
            self.x[collision_mask, 2] = 0

    def simple_ground_collision(self):
        # Simple ground condition for now
        self.x[:, 2].clamp_(min=0)
        self.v[self.x[:, 2] == 0, 2] = 0
