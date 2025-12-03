import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 11  # 每條線的狀態數
ACTIONS = ['left', 'right', 'up', 'down']
EPSILON = 0.9   # The probability of choosing the optimal strategy
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # decay ratio
MAX_EPISODES = 13   # Maximum number of rounds
FRESH_TIME = 0.3    # Travel interval time
CENTER = N_STATES // 2  # 正中心位置

def build_q_table(n_states, actions):
    # 狀態表示: (位置, 方向) - 'h'水平線, 'v'垂直線
    states = []
    for pos in range(n_states):
        states.append(f'h_{pos}')  # 水平線上的位置
        states.append(f'v_{pos}')  # 垂直線上的位置
    table = pd.DataFrame(
        np.zeros((len(states), len(actions))),     
        columns=actions,
        index=states
    )
    return table

tmp_table = build_q_table(N_STATES, ACTIONS)
print(tmp_table)

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.loc[state, :]
    pos = int(state.split('_')[1])
    direction = state.split('_')[0]  # 'h' 或 'v'
    
    # 根據當前方向和位置決定可用動作
    if direction == 'h':
        if pos == CENTER:
            available_actions = ACTIONS  # 在中心可以選擇所有方向
        else:
            available_actions = ['left', 'right']  # 只能左右
    else:  # 垂直線
        if pos == CENTER:
            available_actions = ACTIONS
        else:
            available_actions = ['up', 'down']  # 只能上下
    
    available_q = state_actions[available_actions]
    
    if (np.random.uniform() > EPSILON) or ((available_q == 0).all()):
        action_name = np.random.choice(available_actions)
    else:
        action_name = available_q.idxmax()
    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    direction = S.split('_')[0]
    pos = int(S.split('_')[1])
    R = 0
    
    if A == 'right':
        if pos == N_STATES - 2:  # 水平線右端終點 T
            S_ = 'terminal_h_right'
            R = 1
        else:
            S_ = f'h_{pos + 1}'
    elif A == 'left':
        if pos == 1:  # 水平線左端 c
            S_ = 'terminal_h_left'
            R = -1
        else:
            S_ = f'h_{pos - 1}'
    elif A == 'up':
        if pos == N_STATES - 2:  # 垂直線上端 C
            S_ = 'terminal_v_up'
            R = -1
        else:
            S_ = f'v_{pos + 1}'
    elif A == 'down':
        if pos == 1:  # 垂直線下端 T
            S_ = 'terminal_v_down'
            R = 1
        else:
            S_ = f'v_{pos - 1}'
    
    # 在中心切換方向
    if pos == CENTER and A in ['up', 'down'] and direction == 'h':
        S_ = f'v_{int(S_.split("_")[1])}'
    elif pos == CENTER and A in ['left', 'right'] and direction == 'v':
        S_ = f'h_{int(S_.split("_")[1])}'
    
    return S_, R

def update_env(S, episode, step_counter):
    # 清除螢幕並顯示二維十字型環境
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 初始化十字型網格
    grid_size = N_STATES + 1  # 額外空間給邊界標記
    
    if isinstance(S, str) and S.startswith('terminal'):
        # 顯示結束訊息
        print(f'\nEpisode {episode + 1}: total_steps = {step_counter}')
        print('=' * 30)
        time.sleep(1.5)
    else:
        direction = S.split('_')[0]
        pos = int(S.split('_')[1])
        
        # 建立二維顯示
        # 垂直線: 位置 0 在下方(T), 位置 N_STATES-1 在上方(C)
        # 水平線: 位置 0 在左方(c), 位置 N_STATES-1 在右方(T)
        
        print(f'Episode {episode + 1} | Step: {step_counter}')
        print(f'方向: {"水平" if direction == "h" else "垂直"} | 位置: {pos}')
        print()
        
        # 從上到下繪製
        for row in range(N_STATES):
            v_pos = N_STATES - 1 - row  # 垂直位置 (上方是高索引)
            line = ''
            
            for col in range(N_STATES):
                if col == CENTER and v_pos == CENTER:
                    # 中心點
                    if direction == 'h' and pos == CENTER:
                        line += 'o'  # agent 在水平線中心
                    elif direction == 'v' and pos == CENTER:
                        line += 'o'  # agent 在垂直線中心
                    else:
                        line += '+'  # 交叉點
                elif col == CENTER:
                    # 垂直線上的點 (非中心)
                    if v_pos == N_STATES - 1:
                        if direction == 'v' and pos == N_STATES - 1:
                            line += 'o'
                        else:
                            line += 'C'  # 上方目標
                    elif v_pos == 0:
                        if direction == 'v' and pos == 0:
                            line += 'o'
                        else:
                            line += 'T'  # 下方目標
                    else:
                        if direction == 'v' and pos == v_pos:
                            line += 'o'  # agent 在這裡
                        else:
                            line += '|'
                elif v_pos == CENTER:
                    # 水平線上的點 (非中心)
                    if col == 0:
                        if direction == 'h' and pos == 0:
                            line += 'o'
                        else:
                            line += 'c'  # 左方 (負獎勵)
                    elif col == N_STATES - 1:
                        if direction == 'h' and pos == N_STATES - 1:
                            line += 'o'
                        else:
                            line += 'T'  # 右方目標
                    else:
                        if direction == 'h' and pos == col:
                            line += 'o'  # agent 在這裡
                        else:
                            line += '-'
                else:
                    line += ' '
            
            print(line)
        
        print()
        time.sleep(FRESH_TIME)

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = f'h_{CENTER}'  # 從水平線中心開始
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)            
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if not S_.startswith('terminal'): 
                q_target = R + GAMMA * q_table.loc[S_, :].max()
            else:
                q_target = R
                is_terminated = True    

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  
            S = S_  

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
