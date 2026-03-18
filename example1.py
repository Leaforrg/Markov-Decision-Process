import numpy as np

# 3x3 Grid: 0-7 are paths, 8 is the goal
states = 9
actions = [0, 1, 2, 3] # Up, Down, Left, Right
v_table = np.zeros(states) # Value of each state
gamma = 0.95 # Future reward weight

for i in range(100): # Iteratively refine the value of each square
    new_v = np.copy(v_table)
    for s in range(states - 1): # Exclude the goal state
        # Simplified logic: Assume taking an action leads to a neighbor state
        possible_rewards = [v_table[min(8, s + 1)], v_table[max(0, s - 1)]]
        new_v[s] = 0 + gamma * max(possible_rewards) # 0 is immediate reward
    
    new_v[8] = 10 # Reward for reaching the goal
    v_table = new_v

print("Value of each state (higher = closer to goal):")
print(v_table.reshape(3,3))