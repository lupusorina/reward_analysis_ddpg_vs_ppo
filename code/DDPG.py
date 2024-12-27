import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

set_seed()

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


# Parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--reward_type",
                        type=str,
                        default="dense",
                        help="Type of reward to use (dense or sparse)")
parser.add_argument("--nb_timesteps",
                        type=int,
                        default=100000, 
                        help="Number of timesteps to train the agent")
args = parser.parse_args()

try:
    env = gym.make("Pendulum-v1")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

orig_step = env.step
def custom_reward_step(self, action, type='dense'):
    assert type in ['dense', 'sparse']
    states, rewards, terminated, truncated, infos = orig_step(action)
    cos_theta, sin_theta, theta_dot = states[:, 0], states[:, 1], states[:, 2]
    theta = torch.atan2(sin_theta, cos_theta)
    action = action.squeeze()
    if type == 'dense':
        rewards = - (theta**2 + 0.1*theta_dot**2 + 0.001*action**2)
    elif type == 'sparse':
        rewards = - (10*torch.tanh(10*theta**2) + 0.1*theta_dot**2 + 0.001*action**2)

    rewards = rewards.reshape(-1, 1)
    return states, rewards, terminated, truncated, infos

env.step = lambda action: custom_reward_step(env, action, type=args.reward_type)

device = env.device

memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)

models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["target_policy"] = Actor(env.observation_space, env.action_space, device)
models["critic"] = Critic(env.observation_space, env.action_space, device)
models["target_critic"] = Critic(env.observation_space, env.action_space, device)

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["batch_size"] = 100
cfg["random_timesteps"] = 100
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/torch/PendulumDDPG_" + str(args.reward_type)

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)

cfg_trainer = {"timesteps": args.nb_timesteps,
               "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

trainer.train()

agent.set_running_mode("eval")
env = gym.make("Pendulum-v1", render_mode="human")
env = wrap_env(env)
states, infos = env.reset()
for _ in range(1000):
    env.render()
    with torch.no_grad():
        actions, _, _ = agent.act(states, _, 0)
        states, rewards, terminated, truncated, infos = env.step(actions)
