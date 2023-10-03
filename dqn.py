import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )
    def forward(self, x):
        return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
env = gym.make('MountainCar-v0',render_mode = "human")

q_network = QNetwork(env).to(device)
target_network = QNetwork(env).to(device)
target_network.load_state_dict(q_network.state_dict())


learning_start = 500
learning_rate = 2e-4
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

total_step = 10000
buffer_size = 10000
batch_size = 128
train_frequency = 10
target_network_frequency = 100
gamma = 0.9
target_update_rate = 1.0
epsilon=0.9
total_loss = 0.

rb = ReplayBuffer(
    buffer_size,
    env.observation_space,
    env.action_space,
    device,
    handle_timeout_termination=False,
)

obs,_= env.reset()


for step in range(total_step):
    
    
    if random.random() > epsilon:
         action = np.array(random.choice([0,1,2]))
    else:
        q_value = q_network(torch.Tensor(obs))
        action = torch.argmax(q_value, dim=0).cpu().numpy()
    next_obs, reward, terminated, truncated, infos = env.step(action)
    if terminated:
        break
    rb.add(obs,next_obs,action,reward,terminated,infos)
    
    env.render()
    obs = next_obs

    if step > learning_start:
        if step % train_frequency == 0:
            data = rb.sample(batch_size)
            with torch.no_grad():
                target_max,_= target_network(data.next_observations).max(dim=1)
                tval = data.rewards.squeeze() + gamma * target_max * (1 - data.dones.squeeze())
            qval = q_network(data.observations).gather(1, data.actions)
            qval = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(tval, qval)
            print(step,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if step % target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    target_update_rate * q_network_param.data + (1.0 - target_update_rate) * target_network_param.data
                )

        


    
