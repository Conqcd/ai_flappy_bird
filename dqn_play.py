from DQN import DQNAgent
from DQN import DQN
import torch
from fbenv import FlappyBirdEnv

def main():
    env = FlappyBirdEnv()
    state = env.reset()
    save_dqn_path = "dqn.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    q_network = torch.load(save_dqn_path, map_location=device, weights_only=False)

    done = False
    score = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        q_values = q_network(state)
        action = q_values.argmax().item()
        next_state, reward, done, _ = env.step(action)
        state = next_state

        if reward > 1:
            score += 1
            print(score)
        # print(f'Return: {sum(reward)}')
    print(score)

if __name__ == '__main__':
    main()