from operator import index
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import math
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######################## Global Variables ########################
episodes = 1
num_iter = 1501
energy = 250
energy_spent = 0
display_interval = 50

# List of outcomes to plot
outcomes = np.zeros(num_iter)
step_count = np.zeros(num_iter)
death_count = np.zeros(num_iter)

######################## Environment Definition ########################
'''
# [ 0] = ⬜"inp_nothing"          **  senses nothing (nothing in the requested cell)
# [ 1] = 🌀"inp_breeze"          **  senses breeze (one cell before a pit)
# [ 2] = 😈"inp_danger"          **  senses danger (current cell has Wumpus or pit - death)
# [ 3] = ✨"inp_flash"           **  senses flash (one cell before gold, sees gold glitter)
# [ 4] = 🏆"inp_goal"            **  senses goal (current cell has gold - reward, which is the goal)
# [ 5] = 🏁"inp_initial"         **  senses start (current cell is starting/exit point)
# [ 6] = ⬛"inp_obstruction"      **  senses obstruction (requested move would result in collision)
# [ 7] = 🦨"inp_stench"          **  senses stench (one cell before a Wumpus)
# [ 8] = ✨"inp_bf"              **  senses breeze/flash (cell 'd' has breeze and flash signals)
# [ 9] = ✨"inp_bfs"             **  senses breeze/flash/stench (cell 'd' has breeze + flash + stench)
# [10] = 🌀"inp_bs"              **  senses breeze/stench (cell 'd' has breeze + stench)
# [11] = ✨"inp_fs"              **  senses flash/stench (cell 'd' has flash + stench)
# [12] = ❌"inp_boundary"        **  hit boundary (forward move resulted in environment boundary collision)

# [ 0] = "out_act_grab"         **  action to grab gold (reward)
# [ 1] = "out_act_leave"        **  action to leave cave (at starting position)
# [ 3] = "out_mov_forward"      **  action to move forward
# [11] = "out_rot_left"         **  action to rotate left {"rotate":["left",2]}=90°; {"rotate":["left",1]}=45°
# [12] = "out_rot_right"        **  action to rotate right {"rotate":["right",2]}=90°; {"rotate":["right",1]}=45°
# [13] = "out_rot_back"         **  action to rotate back {"rotate":["back",0]}={"rotate":["right",4]}=180°'''

baseMap = np.array([
    [0, 7, 0, 1, 2, 1, 0, 0, 1, 0],
    [7, 2, 7, 0, 1, 0, 0, 1, 2, 1],
    [0, 7, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 5, 0, 0, 1, 2, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 10, 0, 0],
    [1, 2, 1, 0, 0, 0, 7, 2, 7, 0],
    [0, 1, 7, 0, 0, 0, 1, 11, 0, 0],
    [0, 7, 2, 7, 0, 1, 2, 4, 3, 0],
    [0, 0, 7, 0, 0, 0, 1, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

directions = ['u', 'r', 'd', 'l']
dir_dic = {'u': (-1, 0), 'd': (1, 0), 'l': (0, -1), 'r': (0, 1)}
reverse_dir_dic = {(-1, 0): 'u', (1, 0): 'd', (0, -1): 'l', (0, 1): 'r'}
person_dict = {'u': '⬆️', 'd': '⬇️', 'l': '⬅️', 'r': '➡️'}
char_vector = ['⬜', '🌀', '😈', '✨', '🏆', '🏁', '⬛', '🦨', '✨', '✨', '🌀', '✨', '❌']
mapped_movements = {0: 'g', 1: 'v', 3: 'f', 11: 'l', 12: 'r', 13: 'b'}

# Function to print current state
def print_state(map, pos, direction):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (i, j) == pos:
                print(person_dict[direction], end='  ')
            else:
                print(char_vector[map[i][j]], end=' ')
        print()

def sense(map, pos, dir=-1):
    """
    Function to sense the environment at a given position and direction
    Returns the value of the cell being sensed
    """
    if dir == -1: return map[pos]

    shape = map.shape
    dir = dir_dic[dir]
    dest = (pos[0] + dir[0], pos[1] + dir[1])

    if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
        return 6

    return map[dest]

def sense_vector(map, pos, dir, orientation):
    """
    Creates a vector of sensory inputs based on the agent's position and orientation
    Returns a vector of binary sensors for each possible direction
    """
    vector = np.zeros((len(orientation), 13), dtype=np.int32)
    for i in range(len(orientation)):
        if orientation[i] == 'f':
            vector[i][sense(map, pos, dir)] = 1
        elif orientation[i] == 'l':
            vector[i][sense(map, pos, directions[(directions.index(dir) - 1) % 4])] = 1
        elif orientation[i] == 'r':
            vector[i][sense(map, pos, directions[(directions.index(dir) + 1) % 4])] = 1
        elif orientation[i] == 'b':
            vector[i][sense(map, pos, directions[(directions.index(dir) + 2) % 4])] = 1
    return vector

def move(map, pos, dir, command, win):
    """
    Executes movement commands and updates agent's position and orientation
    Returns updated position, direction, win status, and death status
    """
    if command == 11:  # Rotate left
        dir = directions[directions.index(dir) - 1]
    elif command == 12:  # Rotate right
        dir = directions[(directions.index(dir) + 1) % 4]
    elif command == 13:  # Rotate back
        dir = directions[(directions.index(dir) + 2) % 4]
    elif command == 3:  # Move forward
        shape = map.shape
        calc_dir = dir_dic[dir]
        dest = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
        if dest[0] >= 0 and dest[1] >= 0 and shape[0] > dest[0] and shape[1] > dest[1]:
            pos = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
            dir = reverse_dir_dic[calc_dir]
    if map[pos] == 4:
        win = True
    return pos, dir, win, map[pos] == 2

###################### Deep Q Learning Implementation ###########################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    Implementation of replay memory for experience replay in DQL
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Deep Q-Network implementation with multiple layers
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 125)
        self.layer2 = nn.Linear(125, 75)
        self.layer3 = nn.Linear(75, 18)
        self.layer4 = nn.Linear(18, 75)
        self.layer5 = nn.Linear(75, 125)
        self.layer6 = nn.Linear(125, n_actions)

    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.layer1(x.to(torch.float32)))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)

# Hyperparameters
BATCH_SIZE = 350
GAMMA = 0.65
EPS_START = 0.98
EPS_END = 0.10
EPS_DECAY = 3*10**4
TAU = 0.000085
LR = 0.00022

def summarize_input(reading):
    """
    Converts raw sensor readings into simplified categories
    Returns an integer representing the summarized input
    """
    summary = 0
    if reading in [0, 5]:  # nothing, none
        summary = 0
    elif reading in [3, 8, 9, 11]:  # any flash signal
        summary = 1
    elif reading in [1, 7, 10]:  # any breeze or stench
        summary = 2
    elif reading in [2, 6, 12, 15]:  # any obstacle
        summary = 3
    elif reading in [4]:  # found goal
        summary = 4
    return summary

def calculate_reward(entry_f, entry_d, entry_e, action):
    """
    Calculates reward based on action and surrounding environment
    Returns the reward value for the given state-action pair
    """
    action_num = int(action)
    reward_value = 0
    # Forward movement
    if action_num == 0:
        if entry_f == 0:  # nothing ahead
            reward_value = 0
        elif entry_f == 1:  # flash signal ahead
            reward_value = 1
        elif entry_f == 2:  # danger signal ahead
            reward_value = -0.01
        elif entry_f == 3:  # obstacle ahead
            reward_value = -2.5
        elif entry_f == 4:  # goal ahead
            reward_value = 15
    # Left movement
    if action_num == 1:
        if entry_e == 0:  # nothing to the left
            reward_value = 0
        elif entry_e == 1:  # flash signal to the left
            reward_value = 0.5
        elif entry_e == 2:  # danger signal to the left
            reward_value = -0.01
        elif entry_e == 3:  # obstacle to the left
            reward_value = -1
        elif entry_e == 4:  # goal to the left
            reward_value = 1.5
    # Right movement
    if action_num == 2:
        if entry_d == 0:  # nothing to the right
            reward_value = 0
        elif entry_d == 1:  # flash signal to the right
            reward_value = 0.5
        elif entry_d == 2:  # danger signal to the right
            reward_value = -0.01
        elif entry_d == 3:  # obstacle to the right
            reward_value = -1
        elif entry_d == 4:  # goal to the right
            reward_value = 1.5
    return reward_value

def train(movements):
    """
    Main training loop for the DQL agent
    Parameters:
        movements: List of possible movement directions ['f', 'l', 'r', 'b']
    """
    global display_interval, state, Transition, episodes, num_iter, energy, outcomes, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, next_state
    
    # Initialize neural networks
    state = (0, 0, 0)
    n_actions = 4  # Number of possible actions
    n_observations = 3  # Number of state observations
    
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(95000)
    
    steps_done = 0
    
    def select_action(state):
        """
        Selects action using epsilon-greedy strategy
        As training progresses, reduces exploration in favor of exploitation
        """
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # Select action with highest expected reward
                return policy_net(state.to(torch.float32)).max(1)[1].view(1, 1)
        else:
            # Random exploration
            return torch.tensor([[random.randint(0, 3)]], device=device, dtype=torch.long)

    def optimize_model():
        """
        Performs one step of optimization on the policy network
        Uses experience replay to update network weights
        """
        if len(memory) < BATCH_SIZE:
            return
            
        # Sample random batch from memory
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute current Q values
        state_action_values = policy_net(state_batch).gather(1, action_batch.to(torch.int64))

        # Compute next state values
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
            
        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    # Main training loop
    for episode in range(num_iter):
        # Initialize environment
        map = np.array(baseMap, copy=True)
        pos = (3, 3)  # Starting position
        direction = random.choice(directions)
        win, grabbed, dead = False, False, False
        energy_spent = 0
        
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = None

        # Episode loop
        for energy_spent in range(energy):
            os.system('cls' if os.name == 'nt' else 'clear')
            vector = sense_vector(map, pos, direction, movements)

            # Process current state
            decisions = [3, 11, 12, 13]  # Forward / Left / Right / Back
            front_reading = np.nonzero(vector[0])[0]
            right_reading = np.nonzero(vector[2])[0]
            left_reading = np.nonzero(vector[1])[0]
            
            # Summarize sensor readings
            summary_front = summarize_input(front_reading)
            summary_right = summarize_input(right_reading)
            summary_left = summarize_input(left_reading)

            state = (summary_front, summary_right, summary_left)
            state = torch.tensor([state], device=device)
            action = select_action(state)

            # Get reward and execute action
            reward = calculate_reward(summary_front, summary_right, summary_left, action)
            reward = torch.tensor([reward], device=device)

            # Execute movement and update state
            command = decisions[int(action.item())]
            pos, direction, win, dead = move(map, pos, direction, command, win)

            # Store transition in memory
            observation = np.array([pos[0] - 3, pos[1] - 3, summarize_input(map[pos])])
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)
            
            # Move to next state and optimize model
            state = next_state
            optimize_model()

            # Update target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            # Handle death and victory conditions
            if dead:
                next_state = None
                death_count[episode] += 1
                pos = (3, 3)
            if win:
                next_state = None
                outcomes[episode] = 1
                break

        # Record episode statistics
        step_count[episode] = float(energy_spent)
        
        # Display progress at intervals
        if episode % display_interval == 0 and episode != 0:
            success_rate = (np.sum(outcomes[np.where(outcomes < episode)]) / (episode+1)) * 100
            avg_steps = (np.sum(step_count[np.where(step_count < episode)]) / (episode+1))
            avg_deaths = (np.sum(death_count[np.where(death_count < episode)]) / (episode+1))
            
            print(f"Episode: {episode}")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Average deaths per episode: {avg_deaths:.2f}")
            print(f"Average steps per episode: {avg_steps:.2f}")

    # Plot training results
    plot_results(outcomes, step_count, death_count)

def plot_results(outcomes, step_count, death_count):
    """
    Plots training results using matplotlib
    Shows victory rate, step count, and death count over episodes
    """
    # Plot victories
    plt.figure(figsize=(12, 5))
    plt.xlabel("Episode number")
    plt.ylabel("Victory")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), outcomes, color="#0A047A", width=1.0)
    plt.show()

    # Plot step counts
    plt.figure(figsize=(12, 5))
    plt.xlabel("Episode number")
    plt.ylabel("Steps taken")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), step_count, color="#0A047A", width=1.0)
    plt.show()

    # Plot death counts
    plt.figure(figsize=(12, 5))
    plt.xlabel("Episode number")
    plt.ylabel("Deaths")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), death_count, color="#0A047A", width=1.0)
    plt.show()

if __name__ == '__main__':
    train(['f', 'l', 'r', 'b'])