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
        return torch.sigmoid(x)

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

def compute_returns(rewards, values, gamma, gae_lambda, masks):
    returns = []
    gae = 0
    last_value = values[-1]
    last_mask = masks[-1]
    for r,v,m in reversed(list(zip(rewards,values,masks))):
        # if(len(returns) == 0):
        #     delta = r
        #     gae = delta - v
        # else:
        delta = r + gamma * last_value * m - v
        gae = delta + gamma * gae_lambda * gae * m
        last_value = v
        last_mask = m
        returns.insert(0, gae + v)
    return returns

def ppo_update(policy_net, value_net, optimizer, states, actions, log_probs, returns, advantages, clip_epsilon=0.2,max_grad_norm=1.0):
    wa = 1
    wv = 1
    we = 0.0001

    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    for _ in range(50):  # Update for 10 epochs
        values = value_net(states)
        value_loss = (returns - values).pow(2).mean()
        print(value_loss.detach().cpu().numpy())

        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

    for _ in range(25):  # Update for 10 epochs
        action_probs = policy_net(states)
        two_probs = torch.stack([action_probs[:,0], 1 - action_probs[:,0]], dim=1)
        dist = Categorical(two_probs)
        new_log_probs = dist.log_prob(actions).unsqueeze(1)
        entropy = dist.entropy().unsqueeze(1).sum(-1).mean()

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = value_net(states)
        value_loss = (returns - values).pow(2).mean()

        optimizer.zero_grad()
        loss = value_loss * wv - entropy * we + policy_loss * wa
        print(policy_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy(), entropy.detach().cpu().numpy(),loss.detach().cpu().numpy())
        loss.backward()
        nn.utils.clip_grad_norm_(
            policy_net.parameters(), max_grad_norm
        )
        optimizer.step()


def main():
    env = FlappyBirdEnv()
    # state = env.reset()
    done = False
    replay_buffer_size = 10000
    use_save = True
    save_actor_path = "actor.pth"
    save_critic_path = "critic.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if use_save:
        policy_net = torch.load(save_actor_path, map_location = device, weights_only=False)
        value_net = torch.load(save_critic_path, map_location = device, weights_only=False)
    else:
        policy_net = PolicyNetwork(action_dim).to(device)
        value_net = ValueNetwork(action_dim).to(device)
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.AdamW(params, lr=1e-4, amsgrad=True,weight_decay=0.001)

    max_episodes = 1000000
    gamma = 0.99
    gae_lambda = 0.99

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards, log_probs, values, probs, masks = [], [], [], [], [], [], []

        done = False
        iter = 0
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                value = value_net(state_tensor)
            two_probs = torch.stack([action_probs[:,0], 1 - action_probs[:,0]], dim=1)
            dist = Categorical(two_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            # print(two_probs.detach().cpu().numpy(),action.detach().cpu().numpy())

            next_state, reward, done, _ = env.step(action.item())
            # print(reward)
            if done:
                masks.append(False)
                env.reset_image()
            else:
                masks.append(True)

            probs.append(two_probs)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

            iter += 1

            if done and iter >= replay_buffer_size:
                break

        returns = compute_returns(rewards, values, gamma, gae_lambda, masks)
        returns = torch.FloatTensor(returns).to(device).unsqueeze(1)
        states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
        actions = torch.LongTensor(actions).to(device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze(1)
        advantages = returns - values

        ppo_update(policy_net, value_net, optimizer, states, actions, log_probs, returns, advantages)

        print(f'Episode {episode}, Return: {sum(rewards)}')
        if episode % 10 == 0:
            # print(f'Episode {episode}, Return: {sum(rewards)}')
            torch.save(policy_net, save_actor_path)
            torch.save(value_net, save_critic_path)

if __name__ == '__main__':
    main()