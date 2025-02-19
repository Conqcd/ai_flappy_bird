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
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1,1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, action_dim):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.pool1 = nn.MaxPool2d(2, 2, padding=1)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1,1600)
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

def ppo_update(policy_net, value_net, optimizer, states, actions, log_probs, returns, advantages, clip_epsilon=0.2):
    wa = 1
    wv = 1
    we = 0.01
    for _ in range(10):  # Update for 10 epochs
        action_probs = policy_net(states)
        two_probs = torch.stack([action_probs[:,0], 1 - action_probs[:,0]], dim=1)
        dist = Categorical(two_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().sum(-1).mean()

        ratio = torch.exp(new_log_probs / log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (returns - value_net(states)).pow(2).mean()


        optimizer.zero_grad()
        (policy_loss * wa + value_loss * wv - entropy * we).backward()
        optimizer.step()


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
        policy_net = PolicyNetwork(action_dim).to(device)
        value_net = ValueNetwork(action_dim).to(device)
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    max_episodes = 10000
    gamma = 0.99

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            two_probs = torch.stack([action_probs[:,0], 1 - action_probs[:,0]], dim=1)
            dist = Categorical(two_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
        actions = torch.LongTensor(actions)
        log_probs = torch.stack(log_probs)
        advantages = returns - value_net(states).detach()

        ppo_update(policy_net, value_net, optimizer, states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            print(f'Episode {episode}, Return: {sum(rewards)}')
            torch.save(policy_net, save_actor_path)
            torch.save(value_net,save_critic_path)

if __name__ == '__main__':
    main()