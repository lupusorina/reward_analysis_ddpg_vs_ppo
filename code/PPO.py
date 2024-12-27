import gym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system.
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

set_seed()

# Define models (stochastic and deterministic models) using mixins.
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


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
    env = gym.vector.make("Pendulum-v1", num_envs=4, asynchronous=False)
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("Pendulum-v")][0]
    print("Pendulum-v1 not found. Trying {}".format(env_id))
    env = gym.vector.make(env_id, num_envs=4, asynchronous=False)
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

# Instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device, clip_actions=True)
models["value"] = Value(env.observation_space, env.action_space, device)
# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = False
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 0.5
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 5000
cfg["experiment"]["directory"] = "runs/torch/PendulumPPO_" + str(args.reward_type)

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": args.nb_timesteps,
               "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# Start training
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
