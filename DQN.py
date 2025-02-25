from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from fbenv import FlappyBirdEnv

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
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
        return self.fc2(x)
class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        use_save = True
        self.save_dqn_path = "dqn.pth"
        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_every = 10
        self.steps = 0
        if use_save:
            self.q_network = torch.load(self.save_dqn_path, map_location=device, weights_only=False)
            self.epsilon = 0.01
            self.epsilon_min = 0.0001
        else:
            self.q_network = DQN(action_dim).to(device)

        self.target_network = DQN(action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-4, amsgrad=True, weight_decay=0.001)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(loss.detach().cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            torch.save(self.q_network, self.save_dqn_path)

def main():
    env = FlappyBirdEnv()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n + 1

    agent = DQNAgent(state_dim, action_dim, device)

    max_episodes = 1000000

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards, log_probs, values, probs = [], [], [], [], [], []

        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            rewards.append(reward)

        agent.replay()
        agent.replay()
        print(f'Episode {episode}, Return: {sum(rewards)}')

if __name__ == '__main__':
    main()