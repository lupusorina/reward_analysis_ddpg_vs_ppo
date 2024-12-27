
import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_cost_sparse(theta, theta_dot):
    cost = 10.0 * torch.tanh(10*theta**2) + 0.1 * theta_dot ** 2
    return cost

def compute_cost_dense(theta, theta_dot):
    cost = 1.0 * theta**2 + 0.1 * theta_dot ** 2
    # Kinetic energy vs. potential energy
    return cost

theta, theta_dot = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
theta_torch = torch.tensor(theta, dtype=torch.float32)
theta_dot_torch = torch.tensor(theta_dot, dtype=torch.float32)
cost_sparse = compute_cost_sparse(theta_torch, theta_dot_torch)
cost_dense = compute_cost_dense(theta_torch, theta_dot_torch)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta, theta_dot, cost_sparse, label='Sparse Reward')
ax.plot_surface(theta, theta_dot, cost_dense, label='Dense Reward')
ax.set_xlabel('Theta')
ax.set_ylabel('Theta_dot')
ax.set_zlabel('Cost')
ax.legend()
# plt.savefig('reward.png')

plt.show()