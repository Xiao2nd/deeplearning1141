import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Hyperparameters
N_STATES = 6
ACTIONS = ['left', 'right']  # 動作名稱
N_ACTIONS = len(ACTIONS)  # 動作數量
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 50
MEMORY_CAPACITY = 200
MAX_EPISODES = 100
FRESH_TIME = 0.5  # 延遲時間（秒）

# DQN Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # 輸入為單一狀態值
        self.out = nn.Linear(20, N_ACTIONS)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.out(x)

# DQN Agent
class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 4))  # s, a, r, s_
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        x = torch.FloatTensor([[s]])  # shape: (1, 1)
        if np.random.uniform() < EPSILON:
            action_idx = torch.max(self.eval_net(x), 1)[1].item()
        else:
            action_idx = np.random.randint(0, N_ACTIONS)
        return ACTIONS[action_idx]  # 回傳 'left' 或 'right'

    def store_transition(self, s, a, r, s_):
        a_idx = ACTIONS.index(a)  # 將動作名稱轉為索引
        transition = np.array([s, a_idx, r, s_])
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, 0:1])   # 狀態
        b_a = torch.LongTensor(b_memory[:, 1:2].astype(int))  # 動作
        b_r = torch.FloatTensor(b_memory[:, 2:3])   # 獎勵
        b_s_ = torch.FloatTensor(b_memory[:, 3:4])  # 下一狀態
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Environment feedback
def get_env_feedback(S, A):
    if A == 'right':  # move right
        if S == N_STATES - 2:
            S_ = -1  # terminal (good)
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # move left
        if S == 1:
            S_ = -2  # bad_terminal
            R = -1
        else:
            S_ = S - 1
            R = 0
    return S_, R

# Update environment display
def update_env(S, episode, step_counter, done_msg=None):
    env_list = ['o'] + ['-']*(N_STATES-2) + ['T']
    if done_msg:
        print('\r' + done_msg, end='')
        time.sleep(1)
        print('\n', end='')
    elif S >= 0:
        env_list[S] = '*'
        print('\r' + ''.join(env_list), end='')
        time.sleep(FRESH_TIME)

# Main RL loop
def rl():
    dqn = DQN()
    for episode in range(MAX_EPISODES):
        S = N_STATES//2  # 從中間開始
        step_counter = 0
        update_env(S, episode, step_counter)
        
        while True:
            A = dqn.choose_action(S)
            S_, R = get_env_feedback(S, A)
            
            # 若為終止狀態，下一狀態設為 0
            s_next = S_ if S_ >= 0 else 0
            dqn.store_transition(S, A, R, s_next)
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
            
            step_counter += 1
            
            if S_ == -1:  # Good terminal
                update_env(S_, episode, step_counter, 
                          f'Episode {episode+1}: steps={step_counter} (reached T, R=1)')
                break
            elif S_ == -2:  # Bad terminal
                update_env(S_, episode, step_counter,
                          f'Episode {episode+1}: steps={step_counter} (reached o, R=-1)')
                break
            else:
                S = S_
                update_env(S, episode, step_counter)
    
    return dqn

# 顯示 Q-Table（從 DQN 網路輸出）
def show_q_table(dqn):
    print('\nQ-Table (from DQN):')
    print(f'State\t| {ACTIONS[0]}\t\t| {ACTIONS[1]}')
    print('-' * 40)
    for s in range(N_STATES):
        x = torch.FloatTensor([[s]])
        q_values = dqn.eval_net(x).detach().numpy()[0]
        print(f'  {s}\t| {q_values[0]:.4f}\t\t| {q_values[1]:.4f}')

if __name__ == "__main__":
    dqn = rl()
    print('\nTraining completed!')
    show_q_table(dqn)
