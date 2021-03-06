#%%
"""
リバーシのDQNの推論コードの実装に際して、以下を参考にさせていただいた。

reference : books and program codes
https://github.com/YutaroOgawa/Deep-Reinforcement-Learning-Book
https://tadaoyamaoka.booth.pm/items/1830557

library for othello game and reinforcement learning environment
https://github.com/TadaoYamaoka/creversi

"""
import gym
import creversi.gym_reversi
from creversi import Board, move_to_str, move_from_str, PASS, BLACK_TURN

import sys
import json
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
# use only CPU
device = torch.device("cpu")

#%%
# Normal CNN
class NormalNetwork(nn.Module):
    """
        AIオセロ構築にあたって、DQNで学習させ、学習済ネットワークを得た。
        このPythonスクリプトでは、その学習済ネットワークをロードして、最適手の推論に利用する。
        その際に利用する学習済ネットワークの構造を表すクラス
    """

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
class GreedyPlayer:
    """
        学習済のネットワークを用いて、最適手の推論を実施するPlayerを表すクラス
    """
    def __init__(self, model_path, device, model_setting):
        network_mode = model_setting['network_mode']
        k = model_setting['k']
        fcl_units = model_setting['fcl_units']
        
        self.device = device
        if network_mode == "NormalNetwork":
            self.model = NormalNetwork(k, fcl_units).to(device)
        elif network_mode == "DuelingNetwork":
            self.model = DuelingNetwork(k, fcl_units).to(device) # 現在は、使用していない。
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.features = np.empty((1, 2, 8, 8), np.float32)

    def go(self, board):
        with torch.no_grad():
            board.piece_planes(self.features[0])
            state = torch.from_numpy(self.features).to(self.device)
            q = self.model(state)

            # 石を置くことができる場所（合法手）に絞る
            legal_moves = list(board.legal_moves)
            next_actions = torch.tensor([legal_moves], device=self.device, dtype=torch.long)
            legal_q = q.gather(1, next_actions)

            return legal_moves[legal_q.argmax(dim=1).item()]


class RandomPlayer:
    """
        石を置くことができる場所のうち、ランダムに手を選ぶPlayerを表すクラス
        現在は、使用していない。
    """
    def go(self, board):
        legal_moves = board.legal_moves
        if len(legal_moves) == 0:
            return PASS
        else:
            return random.choice(list(legal_moves))


#%%

def get_board_status_str(board_array):
    """
        2次元numpy.arrayで得た盤面情報を、stringに変換する。
    """
    board_array_1d = board_array.flatten().astype(str)
    board_status_str = ''.join(board_array_1d).replace('0', '-').replace('1','X').replace('2','O')
    # '------------------OOO------OXX----OOXX----OX--------------------'

    return board_status_str


#%%
def get_board_array_revised(board_array):
    """
        2次元numpy.arrayの8*8の盤面に変換する。
        ・stringの場合には、一旦numpy.arrayに変換する。
        ・10*10の場合には、外周の32マスが含まれてしまっているので、外周を取り除く。
    """

    board_array_type_str = str(type(board_array))
    print(board_array_type_str, file=sys.stdout)

    # 受領したデータがstring型であるか確認する。
    if  board_array_type_str == "<class 'str'>":
        # stringの場合には、一旦numpy.arrayに変換する。
        print(board_array_type_str, file=sys.stdout)
        board_array = np.array(json.loads(board_array))
        print(board_array, file=sys.stdout)
        print(str(type(board_array)), file=sys.stdout)

    board_array = np.array(board_array) # 念の為、numpy.arrayに変換している。
    board_array_shape_str = str(board_array.shape)

    # 10*10の場合には、外周の32マスが含まれてしまっているので、外周を取り除く。
    if board_array_shape_str=="(10, 10)":
        board_array_revised = board_array[1:9, 1:9]
    else:
        board_array_revised = board_array

    return board_array_revised
    
#%%
def get_board(board_array, next_player):
    """
        現在の盤面情報や、黒番・白番の情報で初期化した、Boardインスタンスを得る。
    """

    #(10, 10)を(8, 8)に変形する。
    board_array_revised = get_board_array_revised(board_array)

    # boardの状態を適切な記号列に変換する。
    board_status_str = get_board_status_str(board_array_revised)

    # boardの初期化
    # なお、現時点では、AIオセロが黒番の場合しか、AIオセロを利用していない。
    if next_player == 1:
        # 黒番のとき 
        board = Board(board_status_str, creversi.BLACK_TURN)
    else:
        # 白番のとき
        board = Board(board_status_str, creversi.WHITE_TURN)
    
    return board

#%%
def get_move_row_column(move):
    """
        最適手を表す場所(マス目)の情報を、数値2桁で表すように変換する。
    """
    move_str = creversi.move_to_str(move)
    move_row_column = move_str[::-1].replace('a', '1').replace('b','2').replace('c','3').replace('d', '4').replace('e','5').replace('f','6').replace('g', '7').replace('h','8')

    return move_row_column

#%%
def get_greedy_player():
    """
        学習済モデルをロードして準が完了した状態の
        GreedyPlayerインスタンスを作成する。
    """
    model = '/home/shunyu/Code/othello/othello_reinforcement_learning/NormalNetwork_192_256_ExperienceReplay_FixedTargetQ-Network_model2000.pt'
    model_setting = {}
    model_setting['network_mode'] = 'NormalNetwork'
    model_setting['k'] = 192
    model_setting['fcl_units'] = 256 
    greedy_player = GreedyPlayer(model, device, model_setting)

    return greedy_player

#%%
def predict_best_move_row_column_basis(board):
    """
        現在のオセロ盤の情報、黒番・白番の情報(boardインスタンス)をもとに、最適手を得る。
    """

    # 学習済みモデルをロードし、Playerインスタンスを取得する
    ##! 毎回学習済モデルをロードしている点について、現在はパフォーマンス上の問題は出ていないものの、今後改善すべき。
    greedy_player = get_greedy_player()
    # 学習済みモデルで予測する(Playerインスタンスに依頼する)
    best_move = greedy_player.go(board)
    # 予測結果を行・列(数値2桁)形式で表す。
    best_move_row_column_basis = get_move_row_column(best_move)

    return best_move_row_column_basis

#%%
def get_best_move(board_array, next_player):
    """
        現在のオセロ盤の情報、黒番・白番の情報をもとに、最適手を得る。
        まず、現在のオセロ盤の情報、黒番・白番の情報からboardインスタンスを生成し、
        boardインスタンスをもとに最適手を得る。
    """

    # # boardの状態をHTTP経由で与える
    # board_array = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 2, 0, 0],
    #     [0, 0, 0, 1, 2, 0, 0, 0],
    #     [0, 0, 0, 2, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ])
    # # 次の番がどちらであるかHTTP経由で与える
    # next_player = "BLACK"

    # 現在のオセロ盤の情報、黒番・白番の情報からboardインスタンスを生成する
    board = get_board(board_array, next_player)

    # best_moveを得る
    best_move_row_column_basis = predict_best_move_row_column_basis(board)

    return best_move_row_column_basis


#%%
##! 以下は動作の確認用なので、不要。

# boardの状態をHTTP経由で与える
board_array = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 0, 0],
    [0, 0, 0, 1, 2, 0, 0, 0],
    [0, 0, 0, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
# 次の番がどちらであるかHTTP経由で与える
next_player = "BLACK"
get_best_move(board_array, next_player)


# %%
