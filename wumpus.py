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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
######################## Globais ########################
episodes = 1
num_iter = 1001
energia = 250
energia_gasta = 0
intervalo_display = 100

# List of outcomes to plot
outcomes = np.zeros(num_iter)
numero_passos = np.zeros(num_iter)
numero_mortes = np.zeros(num_iter)

######################## Defiunicao do ambiente ########################
'''
# [ 0] = ⬜"inp_nothing"          **  sente nada (tem nada na casa requisitada)
# [ 1] = 🌀"inp_breeze"           **  sente brisa (uma casa antes de um buraco)
# [ 2] = 😈"inp_danger"           **  sente perigo (casa requisitada/atual tem um Wumpus ou um buraco - morre)
# [ 3] = ✨"inp_flash"            **  sente flash (uma casa antes do ouro ele vê o brilho do ouro)
# [ 4] = 🏆"inp_goal"             **  sente meta (casa requisitada/atual tem ouro - reward, que é a meta)
# [ 5] = 🏁"inp_initial"          **  sente início (casa requisitada/atual é o ponto de partida/saída)
# [ 6] = ⬛"inp_obstruction"      **  sente obstução (mandou request,d e vem obstrução é porque vai colidir em 'd')
# [ 7] = 🦨"inp_stench"           **  sente fedor (uma casa antes de um Wumpus)
# [ 8] = ✨"inp_bf"               **  sente brisa/flash (na casa 'd' tem sinais de brisa e flash)
# [ 9] = ✨"inp_bfs"              **  sente brisa/flash/stench (na casa 'd' tem brisa + flash + fedor)
# [10] = 🌀"inp_bs"               **  sente brisa/stench (na casa 'd' tem brisa + fedor)
# [11] = ✨"inp_fs"               **  sente flash/stench (na casa 'd' tem flash + fedor)
# [12] = ❌"inp_boundary"         **  colidiu com borda (mandou mover forward,d e colidiu com a borda do EnviSim)

# [ 0] = "out_act_grab"         **  ação de pegar/agarrar o ouro (reward)
# [ 1] = "out_act_leave"        **  ação de deixar a caverna (no mesmo local de partida)
# [ 3] = "out_mov_forward"      **  ação de mover adiante
# [11] = "out_rot_left"         **  ação de rotacionar esq.{"rotate":["left",2]}=90°; {"rotate":["left",1]}=45°
# [12] = "out_rot_right"        **  ação de rotacionar esq.{"rotate":["right",2]}=90°; {"rotate":["right",1]}=45°
# [13] = "out_rot_back"         **  ação de rotacionar back.{"rotate":["back",0]}={"rotate":["right",4]}=180°'''

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


def print_state(map, pos, direction):
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if (i, j) == pos:
                print(person_dict[direction], end='  ')
            else:
                print(char_vector[map[i][j]], end=' ')
        print()


def sense(map, pos, dir=-1):
    if dir == -1: return map[pos]

    shape = map.shape
    dir = dir_dic[dir]
    dest = (pos[0] + dir[0], pos[1] + dir[1])

    if dest[0] < 0 or dest[1] < 0 or shape[0] <= dest[0] or shape[1] <= dest[1]:
        return 6

    return map[dest]


def senseVector(map, pos, dir, orientation):
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
    if command == 11:
        dir = directions[directions.index(dir) - 1]
    elif command == 12:
        dir = directions[(directions.index(dir) + 1) % 4]
    elif command == 13:
        dir = directions[(directions.index(dir) + 2) % 4]
    elif command == 3:
        shape = map.shape
        calc_dir = dir_dic[dir]
        dest = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
        if dest[0] >= 0 and dest[1] >= 0 and shape[0] > dest[0] and shape[1] > dest[1]:
            pos = (pos[0] + calc_dir[0], pos[1] + calc_dir[1])
            dir = reverse_dir_dic[calc_dir];
    if map[pos] == 4:
            win = True
    return pos, dir, win, map[pos] == 2

###################### Deep Q Learning ###########################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

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

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 200)
        self.layer2 = nn.Linear(200, 18)
        self.layer3 = nn.Linear(18, 200)
        self.layer4 = nn.Linear(200, n_actions)

# Called with either one element to determine next action, or a batch
# during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x.to(torch.float32)))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# Hyperparameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 75
GAMMA = 0.9
EPS_START = 0.98
EPS_END = 0.25
EPS_DECAY = 2000
TAU = 0.005
LR = 0.0006

state = (0, 0, 0)
# Get number of actions from gym action space
n_actions = 4
# Get the number of state observations
n_observations = 3

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)

steps_done = 0


def select_action(state):

    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.to(torch.float32)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 3)]], device=device, dtype=torch.long)

def optimize_model():
    global BATCH_SIZE, GAMMA, Transition
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.to(torch.int64))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def resume_entrada(leitura):
    resum_entry = 0
    if leitura in [0, 5]:  # nothing, none
        resum_entry = 0
    elif leitura in [3, 8, 9, 11]:  # qq um que tenha flash
        resum_entry = 1
    elif leitura in [1, 7, 10]:  # qq um que tenha breeze ou stench
        resum_entry = 2
    elif leitura in [2, 6, 12, 15]:  # qq um que tenha obstáculo
        resum_entry = 3
    elif leitura in [4]:  # encontrou goal
        resum_entry = 4
    return resum_entry


def reward_function(entry_f, entry_d, entry_e, action):
    acao = int(action)
    reward_value = 0
    # ir para frente
    if acao == 0:
        if entry_f == 0:  # nothing, none
            reward_value = 0
        elif entry_f == 1:  # qq um que tenha flash a frente
            reward_value = 1
        elif entry_f == 2:  # qq um que tenha breeze ou stench
            reward_value = -0.01
        elif entry_f == 3:  # qq um que tenha obstáculo
            reward_value = -2.5
        elif entry_f == 4:  # encontrou goal
            reward_value = 15
    # ir para esquerda
    if acao == 1:
        if entry_e == 0:  # nothing, none
            reward_value = 0
        elif entry_e == 1:  # qq um que tenha flash a esquerda
            reward_value = 0.5
        elif entry_e == 2:  # qq um que tenha breeze ou stench
            reward_value = -0.01
        elif entry_e == 3:  # qq um que tenha obstáculo
            reward_value = -1
        elif entry_e == 4:  # encontrou goal
            reward_value = 1.5
    # ir para direita
    if acao == 2:
        if entry_d == 0:  # nothing, none
            reward_value = 0
        elif entry_d == 1:  # qq um que tenha flash a direita
            reward_value = 0.5
        elif entry_d == 2:  # qq um que tenha breeze ou stench
            reward_value = -0.01
        elif entry_d == 3:  # qq um que tenha obstáculo
            reward_value = -1
        elif entry_d == 4:  # encontrou goal
            reward_value = 1.5
    return reward_value


def treino(movements):
    global intervalo_display, state, Transition, episodes, num_iter, energia, comb_dict, outcomes, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, next_state
    # Treinamento
    for episodes in range(num_iter):
        # inicia o ambiente
        map = np.array(baseMap, copy=True)
        pos = (3, 3)
        dir = random.choice(directions)
        win, grabbed, dead = False, False, False
        energia_gasta = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = None
        # realiza cada iteracao do jogo
        for energia_gasta in range(energia):
            os.system('cls' if os.name == 'nt' else 'clear')
            vector = senseVector(map, pos, dir, movements)

            ####################### DQN #######################
            # Resumir e tratar entradas para reduzir complexidade
            decisao = [3, 11, 12, 13]  # Frente / Esquerda / Direita / Tras
            leitura_frente = np.nonzero(vector[0])[0]
            leitura_direita = np.nonzero(vector[2])[0]
            leitura_esquerda = np.nonzero(vector[1])[0]
            resum_entry_f = resume_entrada(leitura_frente)
            resum_entry_d = resume_entrada(leitura_direita)
            resum_entry_e = resume_entrada(leitura_esquerda)

            state = (resum_entry_f, resum_entry_d, resum_entry_e)
            state = torch.tensor([state], device=device)
            action = torch.tensor([select_action(state)], dtype=torch.float32, device=device).unsqueeze(0)

            observation = np.array([pos[0] - 3, pos[1] - 3, resume_entrada(map[pos])])
            # reward
            reward = reward_function(resum_entry_f, resum_entry_d, resum_entry_e, action)
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            command = decisao[int(action.item())]
            pos, dir, win, dead = move(map, pos, dir, command, win)

            # Prints para debug
            """print('Energia:',energia_gasta)
            print("entrada Frente:", resum_entry_f)
            print("entrada direita:", resum_entry_d)
            print("entrada Esquerda:", resum_entry_e)
            print("observation:", observation)
            print("action:", int(action))
            print("Reward:", int(reward))
            print_state(map, pos, dir)
            time.sleep(0.1)"""

            if dead:
                next_state = None
                numero_mortes[episodes] = numero_mortes[episodes] + 1
                #time.sleep(1)
                pos = (3, 3)
            # Encontrou o ouro
            if win:
                next_state = None
                outcomes[episodes] = 1
                #time.sleep(1)
                break
            # Mostra a ultima iteracao
            if episodes > num_iter - 2:
                print('Episodeos:', episodes)
                print_state(map, pos, dir)
                time.sleep(0.1)

        print('episodes: ', episodes)
        numero_passos[episodes] = float(energia_gasta)
        # Printes de controle e save de vetores np
        if episodes % intervalo_display == 0 and episodes != 0:
            # Calculo das porcentagens
            pss_sucesso = (np.sum(outcomes[np.where(outcomes < episodes)]) / (episodes+1)) * 100
            num_passos_med = (np.sum(numero_passos[np.where(numero_passos < episodes)]) / (episodes+1))
            num_mortes_med = (np.sum(numero_mortes[np.where(numero_mortes < episodes)]) / (episodes+1))
            print(f"Porcentagem de sucesso: {pss_sucesso:.2f}%")
            print(f"Numero medio de mortes: {num_mortes_med:.2f}")
            print(f"Numero medio de passos: {num_passos_med:.2f}")

    # Prints de final de treino
    # Vitoria
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Outcome")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), outcomes, color="#0A047A", width=1.0)
    plt.show()

    #Grafico numero de passos
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("Numero de passos")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), numero_passos, color="#0A047A", width=1.0)
    plt.show()

    # Mortes
    plt.figure(figsize=(12, 5))
    plt.xlabel("Run number")
    plt.ylabel("numero_mortes")
    ax = plt.gca()
    ax.set_facecolor('#efeeea')
    plt.bar(range(num_iter), numero_mortes, color="#0A047A", width=1.0)
    plt.show()

if __name__ == '__main__':
    treino(['f', 'l', 'r', 'b'])
