import torch
import torch.nn as nn


# Differentialable Spring-Mass Simulator
class SpringMass(nn.Module):
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
        drag_damping=1,
    ):
        super().__init__()
        # Number of mass and springs
        self.n_vertives = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]
        # Initialization
        self.x = init_vertices
        self.v = torch.zeros(self.n_vertives)
        self.springs = init_springs
        self.rest_lengths = init_rest_lengths
        self.masses = init_masses
        # Internal forces
        self.spring_forces = None
        self.vertice_forces = torch.zeros((self.n_vertives, 3), requires_grad=True)

        self.dt = dt
        self.num_substeps = num_substeps
        self.spring_Y = spring_Y
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping

    def step(self):
        for i in range(self.num_substeps):
            self.substep()

        return (
            self.x,
            self.springs,
            self.rest_lengths,
            self.spring_forces,
        )

    def substep(self):
        # One simulation step of the spring-mass system
        self.vertice_forces.zero_()

        # Add teh gravity force
        self.vertice_forces += self.masses * torch.tensor([0.0, 0.0, -9.8])
        # Calculate the spring forces
        idx1 = self.springs[:, 0]
        idx2 = self.springs[:, 1]
        x1 = self.x[self.springs[:, 0]]
        x2 = self.x[self.springs[:, 1]]
        dis = x2 - x1
        d = dis / torch.norm(dis, dim=1)[:, None]
        self.spring_forces = (
            self.spring_Y
            * (torch.norm(dis, dim=1) / self.rest_lengths - 1)[:, None]
            * d
        )

        self.forces[idx1] += self.spring_forces
        self.forces[idx2] -= self.spring_forces

        # Apply the damping forces
        v_rel = torch.einsum("ij,j->i", (self.v[idx2] - self.v[idx1]), d)
        dashpot_forces = self.dashpot_damping * v_rel[:, None] * d
        self.forces[idx1] += dashpot_forces
        self.forces[idx2] -= dashpot_forces

        # Update the velocity
        self.v += self.dt * self.forces / self.masses[:, None]
        self.v *= torch.exp(-self.dt * self.drag_damping)
        self.x += self.dt * self.v

        # Simple ground condition for now
        self.x[:, 2].clamp_(min=0)
        self.v[self.x[:, 2] == 0, 2] = 0
