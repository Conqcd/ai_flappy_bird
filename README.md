# flappy_bird_ai

用了三种实现方法: 1. Neat 2. PPO 3. DQN

其中DQN表现最好，PPO表现最差，DQN算法最好能够训练到2000分左右，PPO在没有魔改reward的情况反复在1到2分的局部最优解，而Neat只训练到150分左右。