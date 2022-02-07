# -*- coding: utf-8 -*-
"""
リバーシのDQN実装

reference : books and program codes
https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
https://tadaoyamaoka.booth.pm/items/1830557

library for osero game and reinforcement learning environment
https://github.com/TadaoYamaoka/creversi

"""
import gym
import creversi.gym_reversi
from creversi import Board, move_to_str, move_from_str, PASS, BLACK_TURN

import os
import datetime
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%%


# setting
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
OPTIMIZE_PER_EPISODES = 16
TARGET_UPDATE = 4
TD_ERROR_EPSILON = 0.0001
CAPACITY = 10000

# use only CPU
device = torch.device("cpu")


#%%

"""
経験データ(transition)のタプルをnamedtupleとして定義します。
通常、経験データは{現在の状態, 選択した行動, 次の状態, 報酬}の4つですが、効率化のために次の状態(局面)の合法手の一覧も格納するようにします。

define ReplayMemory management class.
define TDErrorMemory management class, which is used for Prioritized Experience Replay.
"""

######################################################################
# define transition tuple as namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'next_actions', 'reward'))


class ReplayMemory(object):

    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TDErrorMemory(object):
    
    def __init__(self):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0
        
    def push(self, td_error):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity
        
    def __len__(self):
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):
        
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)
        
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)
        
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1
                
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)
        
        return indexes
    
    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors
#%%
"""
ニューラルネットワークを定義します。10層の畳み込みニューラルネットワークに全結合層を接続します。
出力の活性化関数はtanhとして行動価値を-1～1の範囲で出力します。
"""

######################################################################
# DQN
# Normal CNN
class NormalNetwork(nn.Module):

    def __init__(self, k, fcl_units):
        super(NormalNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(2, k, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.conv2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(k)
        self.conv3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(k)
        self.conv4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(k)
        self.conv5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(k)
        self.conv6 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(k)
        self.conv7 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(k)
        self.conv8 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(k)
        self.conv9 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(k)
        self.conv10 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(k)
        self.fcl1 = nn.Linear(k * 64, fcl_units)
        self.fcl2 = nn.Linear(fcl_units, 65)
        
        # size of channels
        self.k = k


    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.fcl1(x.view(-1, self.k * 64)))
        x = self.fcl2(x)
        y = x.tanh()
        return y

#%%
# Dueling Network
class DuelingNetwork(nn.Module):

    def __init__(self, k, fcl_units):
        super(DuelingNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(2, k, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(k)
        self.conv2 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(k)
        self.conv3 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(k)
        self.conv4 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(k)
        self.conv5 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(k)
        self.conv6 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(k)
        self.conv7 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(k)
        self.conv8 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(k)
        self.conv9 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(k)
        self.conv10 = nn.Conv2d(k, k, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(k)
        self.fcl1 = nn.Linear(k * 64, fcl_units)
        
        # Dueling Network
        self.fcl2_adv = nn.Linear(fcl_units, 65)
        self.fcl2_v = nn.Linear(fcl_units, 1)
        
        # size of channels
        self.k = k

    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.fcl1(x.view(-1, self.k * 64)))
        
        adv = self.fcl2_adv(x)
        val = self.fcl2_v(x).expand(-1, adv.size(1))
        
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        
        y = output.tanh()

        return y

#%%
# ResNet

class Block(nn.Module):

    def __init__(self, channel_in, channel_out):
        super(Block, self).__init__()
        channel = channel_out

        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(channel, channel_out, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)

        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


class GloblAvgPool2d(nn.Module):
    def __init__(self, device='cpu'):
        super(GloblAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))

class ResNet50(nn.Module):
    # def __init__(self, output_dim):
    def __init__(self):
        super(ResNet50, self).__init__()

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=1)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([self._building_block(256) for _ in range(2)])
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))

        self.block2 = nn.ModuleList([self._building_block(512) for _ in range(4)])
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))

        self.block3 = nn.ModuleList([self._building_block(1024) for _ in range(6)])
        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))

        self.block4 = nn.ModuleList([self._building_block(2048) for _ in range(3)])

        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        # self.out = nn.Linear(1000, output_dim)
        self.out = nn.Linear(1000, 65)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        h = torch.relu()
        h = self.out(h)
        # y = torch.log_softmax(h, dim=-1)

        # return y
        y = h.tanh()
        
        return y 

    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)   


#%%
class Brain:
    
    def __init__(self, network_mode, k, fcl_units, batch_sampling_mode, dqn_mode):

        
        self.network_mode = network_mode       
        if network_mode == "NormalNetwork":
            self.policy_net = NormalNetwork(k, fcl_units).to(device)
            self.target_net = NormalNetwork(k, fcl_units).to(device)
        elif network_mode == "DuelingNetwork":
            self.policy_net = DuelingNetwork(k, fcl_units).to(device)
            self.target_net = DuelingNetwork(k, fcl_units).to(device)
            
        #copy from policy_net to target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-5)
        
         
        self.memory = ReplayMemory()       
        self.batch_sampling_mode = batch_sampling_mode
        if batch_sampling_mode == "PrioritizedExperienceReplay":
            self.td_error_memory = TDErrorMemory()
            
        self.dqn_mode = dqn_mode
            
        self.losses = []

    def epsilon_greedy(self, state, legal_moves, episodes_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * episodes_done / EPS_DECAY)
    
        if sample > eps_threshold:
            self.policy_net.eval()
            with torch.no_grad():
                q = self.policy_net(state)
                _, select = q[0, legal_moves].max(0)
        else:
            select = random.randrange(len(legal_moves))
        return select
    
    def select_action(self, state, board, episodes_done):
    
        legal_moves = list(board.legal_moves)
    
        select = self.epsilon_greedy(state, legal_moves, episodes_done)
        
        return legal_moves[select], torch.tensor([[legal_moves[select]]], device=device, dtype=torch.long)

    def make_minibatch(self, i_episode):
        
        
        
        if self.batch_sampling_mode == "ExperienceReplay":
            transitions = self.memory.sample(BATCH_SIZE)
        elif self.batch_sampling_mode == "PrioritizedExperienceReplay":
            if i_episode < 2*OPTIMIZE_PER_EPISODES:
                transitions = self.memory.sample(BATCH_SIZE)
            else:
                indexes = self.td_error_memory.get_prioritized_indexes(BATCH_SIZE)
                transitions = [self.memory.memory[n] for n in indexes]
        
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # 合法手のみ
        non_final_next_actions_list = []
        for next_actions in batch.next_actions:
            if next_actions is not None:
                non_final_next_actions_list.append(next_actions + [next_actions[0]] * (30 - len(next_actions)))
        non_final_next_actions = torch.tensor(non_final_next_actions_list, device=device, dtype=torch.long)
        
        return batch, state_batch, action_batch, reward_batch, non_final_next_states, non_final_next_actions

    def get_expected_state_action_values(self):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        self.policy_net.eval()
        self.state_action_values = self.policy_net(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              self.batch.next_state)), device=device, dtype=torch.bool)        
        
        if self.dqn_mode == "FixedTargetQ-Network":
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                # 合法手のみの最大値
                target_q = self.target_net(self.non_final_next_states)
                # 相手番の価値のため反転する
                next_state_values[non_final_mask] = -target_q.gather(1, self.non_final_next_actions).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = next_state_values * GAMMA + self.reward_batch
                
        elif self.dqn_mode == "DDQN":
            with torch.no_grad():
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                policy_q = self.policy_net(self.non_final_next_states)
                non_final_next_action_choices = policy_q.gather(1, self.non_final_next_actions).max(1)[1].detach().view(-1,1)
                target_q = self.target_net(self.non_final_next_states)
                next_state_values[non_final_mask] = -target_q.gather(1, non_final_next_action_choices).detach().squeeze()
                expected_state_action_values = next_state_values * GAMMA + self.reward_batch
                
        return expected_state_action_values
    def update_policy_net(self):
        
        self.policy_net.train()
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
    
        self.losses.append(loss.item())
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()        
        
        return
        
    def optimize_model(self, i_episode):
        
        if len(self.memory) < BATCH_SIZE:
            return
       
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states, self.non_final_next_actions = self.make_minibatch(i_episode)
        
        self.expected_state_action_values = self.get_expected_state_action_values()
        
        self.update_policy_net()
        
        return

    def update_td_error_memory(self):
        
        start_time_of_update_td_error_memory = time.time()
        
        self.policy_net.eval()

        with torch.no_grad():
            
            # get all transitions
            transitions = self.memory.memory
            batch = Transition(*zip(*transitions))
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            # 合法手のみ
            non_final_next_actions_list = []
            for next_actions in batch.next_actions:
                if next_actions is not None:
                    non_final_next_actions_list.append(next_actions + [next_actions[0]] * (30 - len(next_actions)))
            non_final_next_actions = torch.tensor(non_final_next_actions_list, device=device, dtype=torch.long) 
            
         
            state_action_values = self.policy_net(state_batch).gather(1, action_batch).detach().squeeze()

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool) 
            
            if self.dqn_mode == "FixedTargetQ-Network":
                next_state_values = torch.zeros(len(self.memory), device=device)
                # 合法手のみの最大値
                target_q = self.target_net(non_final_next_states)
                # 相手番の価値のため反転する
                next_state_values[non_final_mask] = -target_q.gather(1, non_final_next_actions).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = next_state_values * GAMMA + reward_batch

            elif self.dqn_mode == "DDQN":
                next_state_values = torch.zeros(len(self.memory), device=device)    
                policy_q = self.policy_net(non_final_next_states)
                non_final_next_action_choices = policy_q.gather(1, non_final_next_actions).max(1)[1].detach().view(-1,1)        
                target_q = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = -target_q.gather(1, non_final_next_action_choices).detach().squeeze()        
                expected_state_action_values = next_state_values * GAMMA + reward_batch
            
                
            td_errors = expected_state_action_values - state_action_values
            
            self.td_error_memory.memory = td_errors.numpy().tolist()
            
        elapsed_time_of_update_td_error_memory = time.time() - start_time_of_update_td_error_memory
            
        return elapsed_time_of_update_td_error_memory
    
    def update_target_net(self):
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        return
                    

    
#%% 

class Agent:
    
    def __init__(self, network_mode, k, fcl_units, batch_sampling_mode, dqn_mode):
        self.brain = Brain(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode)
        self.model_name = network_mode + "_" + str(k) + "_" + str(fcl_units) + "_" + batch_sampling_mode + "_" + dqn_mode
        
    def update_q_function(self, i_episode):
        self.brain.optimize_model(i_episode)
        
    def get_action(self, state, board, episodes_done):
        action = self.brain.select_action(state, board, episodes_done)
        return action
    
    def memorize(self, state, action, next_state, next_actions, reward):
        self.brain.memory.push(state, action, next_state, next_actions, reward)
        
    def update_target_q_function(self):
        self.brain.update_target_net()
        
    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.push(td_error)
        
    def update_td_error_memory(self):
        elapsed_time_of_update_td_error_memory = self.brain.update_td_error_memory()
        return elapsed_time_of_update_td_error_memory
    
    def get_losses(self):
        losses = self.brain.losses
        return losses
    
    def save_model(self, num_episodes):
             
        modelfile = self.model_name + '_model'+str(num_episodes)+'.pt'
        print('save {}'.format(modelfile))
        torch.save({'state_dict': self.brain.target_net.state_dict(), 'optimizer': self.brain.optimizer.state_dict()}, modelfile)
        
        return

#%%
class Environment:
    
    def __init__(self, network_mode, k, fcl_units, batch_sampling_mode, dqn_mode):
        
        # make gym enviroment
        self.env = gym.make('Reversi-v0').unwrapped
        
        self.agent = Agent(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode)
        self.batch_sampling_mode = batch_sampling_mode
        
    def get_state(self, board):
        
        features = np.empty((1, 2, 8, 8), dtype=np.float32)
        board.piece_planes(features[0])
        state = torch.from_numpy(features[:1]).to(device)
        
        return state

    def run(self, num_episodes):
        
        start_time = time.time()

        episodes_done = 0
        pbar = tqdm(total=num_episodes)
        losses = []
        if self.batch_sampling_mode == "PrioritizedExperienceReplay":
            sum_of_elapsed_time_of_update_td_error_memory = 0
        
        
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            state = self.get_state(self.env.board)
            
            for t in count():
                # Select and perform an action
                move, action = self.agent.get_action(state, self.env.board, episodes_done)
                next_board, reward, done, is_draw = self.env.step(move)
        
                reward = torch.tensor([reward], device=device)
        
                # Observe new state
                if not done:
                    next_state = self.get_state(next_board)
                    next_actions = list(next_board.legal_moves)
                else:
                    next_state = None
                    next_actions = None
        
                # Store the transition in memory
                self.agent.memorize(state, action, next_state, next_actions, reward)
                
                # 
                if self.batch_sampling_mode == "PrioritizedExperienceReplay":
                    self.agent.memorize_td_error(0)
        
                if done:
                    break
        
                # Move to the next state
                state = next_state
        
            episodes_done += 1
            pbar.update()
        
            if i_episode % OPTIMIZE_PER_EPISODES == OPTIMIZE_PER_EPISODES - 1:
                # Perform several episodes of the optimization (on the target network)
                self.agent.update_q_function(i_episode)
                
                # update td error memory for prioritied experience replay
                if self.batch_sampling_mode == "PrioritizedExperienceReplay":
                    sum_of_elapsed_time_of_update_td_error_memory += self.agent.update_td_error_memory()
                
                losses = self.agent.get_losses()
                pbar.set_description(f'loss = {losses[-1]:.3e}')
        
                # Update the target network, copying all weights and biases in DQN
                if i_episode // OPTIMIZE_PER_EPISODES % TARGET_UPDATE == 0:
                    self.agent.update_target_q_function()
        self.agent.save_model(num_episodes)
        
        print('Learning Complete')
        self.env.close()
        
        """損失をグラフ表示します。"""
        plt.plot(losses)
        
        elapsed_time = time.time() - start_time
        print("Elapsed Time of Reinforcement Learning:" + str(elapsed_time) + "[sec]")

        if self.batch_sampling_mode == "PrioritizedExperienceReplay":
            print("Elapsed Time of Prioritized Experience Replay:" + str(sum_of_elapsed_time_of_update_td_error_memory) + "[sec]")
        
        return losses

#%%
class GreedyPlayer:
    def __init__(self, model_path, device, model_setting):
        network_mode = model_setting['network_mode']
        k = model_setting['k']
        fcl_units = model_setting['fcl_units']
        
        self.device = device
        if network_mode == "NormalNetwork":
            self.model = NormalNetwork(k, fcl_units).to(device)
        elif network_mode == "DuelingNetwork":
            self.model = DuelingNetwork(k, fcl_units).to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.features = np.empty((1, 2, 8, 8), np.float32)

    def go(self, board):
        with torch.no_grad():
            board.piece_planes(self.features[0])
            state = torch.from_numpy(self.features).to(self.device)
            q = self.model(state)
            # 合法手に絞る
            legal_moves = list(board.legal_moves)
            next_actions = torch.tensor([legal_moves], device=self.device, dtype=torch.long)
            legal_q = q.gather(1, next_actions)
            return legal_moves[legal_q.argmax(dim=1).item()]

class RandomPlayer:
    def go(self, board):
        legal_moves = board.legal_moves
        if len(legal_moves) == 0:
            return PASS
        else:
            return random.choice(list(legal_moves))
#%%
def battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=100):

    players = []
    for player, model, model_setting in zip([player1, player2], [model1, model2], [model1_setting, model2_setting]):
        if player == 'random':
            players.append(RandomPlayer())
        elif player == 'greedy':
            k = 192
            fcl_units = 256
            players.append(GreedyPlayer(model, device, model_setting))

    black_won_count = 0
    white_won_count = 0
    draw_count = 0
    board = Board()
    for n in range(games):
        # print(f'game {n}')
        board.reset()
        move = None

        i = 0
        while not board.is_game_over():
            i += 1

            if board.puttable_num() == 0:
                move = PASS
            else:
                player = players[(i - 1) % 2]
                move = player.go(board)
                assert board.is_legal(move)

            board.move(move)

        if board.turn == BLACK_TURN:
            piece_nums = [board.piece_num(), board.opponent_piece_num()]
        else:
            piece_nums = [board.opponent_piece_num(), board.piece_num()]

        # print(f'result black={piece_nums[0]} white={piece_nums[1]}')
        if piece_nums[0] > piece_nums[1]:
            # print('black won')
            black_won_count += 1
        elif piece_nums[1] > piece_nums[0]:
            # print('white won')
            white_won_count += 1
        else:
            # print('draw')
            draw_count += 1

    print(f'black:{black_won_count} white:{white_won_count} draw:{draw_count}')
#%%
def moving_average(losses, horizen=10):
    
    losses_revised = np.convolve(losses, np.ones(horizen)/horizen, mode="valid")
    
    return losses_revised                        
#%%
## set DQN algorithm
# network_mode = "NormalNetwork"
# network_mode = "DuelingNetwork"

# batch_sampling_mode = "ExperienceReplay"
# batch_sampling_mode = "PrioritizedExperienceReplay"

# dqn_mode = "FixedTargetQ-Network"
# dqn_mode = "DDQN"

## set network hyperparameters
# k = 192
# fcl_units = 256

## other setting 
# num_episodes = 500


## pattern 1

network_mode = "NormalNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "ExperienceReplay"
dqn_mode = "FixedTargetQ-Network"
num_episodes = 2000

losses_1 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)

#%%
## pattern 2

network_mode = "DuelingNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "ExperienceReplay"
dqn_mode = "FixedTargetQ-Network"
num_episodes = 2000

losses_2 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)

#%%
## pattrn 3

network_mode = "NormalNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "PrioritizedExperienceReplay"
dqn_mode = "FixedTargetQ-Network"
num_episodes = 2000

losses_3 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)

#%%
## pattern 4

network_mode = "NormalNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "ExperienceReplay"
dqn_mode = "DDQN"
num_episodes = 2000

losses_4 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)
#%%
## pattern 5

network_mode = "NormalNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "PrioritizedExperienceReplay"
dqn_mode = "DDQN"
num_episodes = 2000

losses_5 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)

#%%
plt.plot(moving_average(losses_1, 10), color='red', label='pattern 1')
plt.plot(moving_average(losses_2, 10), color='blue', label='pattern 2')
plt.plot(moving_average(losses_3, 10), color='green', label='pattern 3')
plt.plot(moving_average(losses_4, 10), color='yellow', label='pattern 4')
plt.plot(moving_average(losses_5, 10), color='gray', label='pattern 5')
plt.legend()

#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/NormalNetwork_192_256_ExperienceReplay_FixedTargetQ-Network_model2000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'NormalNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)

#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/DuelingNetwork_192_256_ExperienceReplay_FixedTargetQ-Network_model2000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'DuelingNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)

#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/NormalNetwork_192_256_PrioritizedExperienceReplay_FixedTargetQ-Network_model2000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'NormalNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)

#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/NormalNetwork_192_256_ExperienceReplay_DDQN_model2000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'NormalNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)
#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/NormalNetwork_192_256_PrioritizedExperienceReplay_DDQN_model2000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'NormalNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)
#%%
network_mode = "NormalNetwork"
k = 192
fcl_units = 256
batch_sampling_mode = "ExperienceReplay"
dqn_mode = "FixedTargetQ-Network"
num_episodes = 100000

losses_1 = Environment(network_mode, k, fcl_units, batch_sampling_mode, dqn_mode).run(num_episodes)
#%%
plt.plot(moving_average(losses_1, 100), color='red', label='pattern 1')
plt.legend()

#%%
player1 = 'greedy'
player2 = 'random'
model1 = '/home/shunyu/Code/osero/NormalNetwork_192_256_ExperienceReplay_FixedTargetQ-Network_model100000.pt'
model1_setting = {}
model1_setting['network_mode'] = 'NormalNetwork'
model1_setting['k'] = 192
model1_setting['fcl_units'] = 256 
battle(player1, player2, model1, model1_setting, model2=None, model2_setting=None, games=1000)