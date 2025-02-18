import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributions as dist
from fbenv import FlappyBirdEnv

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding_mode= Same)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x.view(-1,1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, action_dim):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1, padding_mode='replicate')
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x.view(-1,1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs, returns, advantages, clip_epsilon=0.2):
    for _ in range(10):  # Update for 10 epochs
        new_log_probs = policy_net(states).gather(1, actions.unsqueeze(1)).log()
        ratio = new_log_probs / log_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - value_net(states)).pow(2).mean()

        categorical_dist = dist.Categorical(probs=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        entropy_categorical = categorical_dist.entropy()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

def main():
    env = FlappyBirdEnv()
    state = env.reset()
    done = False
    use_save = False
    save_actor_path = "actor.pth"
    save_critic_path = "critic.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if use_save:
        policy_net = torch.load(save_actor_path, map_location = device)
        value_net = torch.load(save_critic_path, map_location = device)
    else:
        policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        value_net = ValueNetwork(state_dim).to(device)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    max_episodes = 10000
    gamma = 0.99

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.stack(log_probs)
        advantages = returns - value_net(states).detach()

        ppo_update(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            print(f'Episode {episode}, Return: {sum(rewards)}')

if __name__ == '__main__':
    main()