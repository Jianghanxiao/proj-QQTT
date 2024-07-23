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
        dt=5e-5,
        num_substeps=1000,
        spring_Y=3e4,
        dashpot_damping=100,
        drag_damping=3,
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

        # Simple ground condition for now
        self.x[:, 2].clamp_(min=0)
        self.v[self.x[:, 2] == 0, 2] = 0
