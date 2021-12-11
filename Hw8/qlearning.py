import numpy as np

"""
2 States - A, B

Starting at state A

Gamma = 0.8

Rewards:

R(A, Move) = 0
R(A, Stay) = 1
R(B, Move) = 0
R(B, Stay) = 1

"""

"""
Q 2.1 : Deterministic Greedy Policy
"""
q_table = {'A': {'S': 0, 'M': 0}, 'B': {'S': 0, 'M': 0}}

current_state = 'A'

# Running the greedy deterministic policy for 200 steps
for i in range(200):

    stay_qvalue = q_table[current_state]['S']
    move_qvalue = q_table[current_state]['M']

    # Decide action greedily
    if stay_qvalue > move_qvalue:
        next_action = 'S'
    else:
        next_action = 'M'

    # Get current reward for action performed
    reward = 1 if next_action == 'S' else 0

    # Get next state based on action
    if next_action == 'M':
        next_state = 'A' if current_state == 'B' else 'B'
    else:
        next_state = current_state
    
    # Update the Q-table entry
    q_table[current_state][next_action] = 0.5 * (q_table[current_state][next_action]) + \
        0.5 * (reward + 0.8 * max(q_table[next_state]['S'], q_table[next_state]['M']))
    
    current_state = next_state

# Printing the Q-Table at the end
print(q_table)

"""
Q 2.2 : Epsilon Greedy Policy
"""

q_table = {'A': {'S': 0, 'M': 0}, 'B': {'S': 0, 'M': 0}}

current_state = 'A'
epsilon = 0.5

# Running the epsilon deterministic policy for 200 steps
for i in range(200):

    stay_qvalue = q_table[current_state]['S']
    move_qvalue = q_table[current_state]['M']

    # Decide action greedily
    if stay_qvalue > move_qvalue:
        next_action = 'S'
    else:
        next_action = 'M'

    # Sample from uniform distribution to generate random action
    random_action = 'S' if np.random.uniform(0, 1) > 0.5 else 'M'

    # Sample from uniform distribution to determine to either exploit or explore
    next_action = random_action if np.random.uniform(0, 1) > epsilon else next_action

    # Get current reward for action performed
    reward = 1 if next_action == 'S' else 0

    # Get next state based on action
    if next_action == 'M':
        next_state = 'A' if current_state == 'B' else 'B'
    else:
        next_state = current_state     
    
    # Update the Q-table entry
    q_table[current_state][next_action] = 0.5 * (q_table[current_state][next_action]) + \
        0.5 * (reward + 0.8 * max(q_table[next_state]['S'], q_table[next_state]['M']))
    
    current_state = next_state

# Printing the Q-Table
print(q_table)