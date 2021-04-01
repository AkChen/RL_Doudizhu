# 模拟阶段不带洗牌，也就是说处于明牌状态
import copy
from rlcard.envs import Env
import time
import random
import numpy as np
from math import log,sqrt
from rlcard.games.doudizhu.utils import cards2str,doudizhu_sort_card
from collections import OrderedDict
import functools
# single player tree
ROOT_KEY = 0
ROOT_ACTION = 0
CP_VAL = 0.70710678118655

class MPMCTSTreeNode(object):

    def __init__(self,key,action,player_num:int,legal_actions: list,parent = None):
        self.key = key
        self.action = action
        self.legal_actions = legal_actions  # [player_id][legale_actions]
        self.parent = parent
        self.player_num = player_num
        self.visit_num = [1 for i in range(player_num)]  # set at least once, convenient for divide opt.
        self.reward_sum = [0.0 for i in range(player_num)]
        #
        self.children = dict()  # key:action



class MPMCTSAgent(object):

    def __init__(self,env:Env,emu_num :int, # for simulation
):
        self.env = env
        self.use_raw = False
        self.fake_action_prob = [0.0 for i in range(env.action_num)]
        self.emu_num = emu_num

    # 每次run之前都要初始化env，与外部保持一致


    def get_untried_action(self,node:MPMCTSTreeNode,player_id):

        untried_actions = []

        for legal_action in node.legal_actions[player_id]:
            # action没有扩张或者这个player_id访问的次数为1(初始化次数)
            if (not node.children.__contains__(legal_action)) or (node.children[legal_action].visit_num[player_id] == 1):
                untried_actions.append(legal_action)

        if len(untried_actions) == 0:
            return None
        else:
            return random.sample(untried_actions,1)[0]


    # return expanded child node
    def expand_action_on_node(self,node:MPMCTSTreeNode,action,env:Env):
        # 向前移动
        next_state, next_player_id = env.step(action, False)
        if node.children.__contains__(action): # action存在
            new_node = node.children[action]
            new_node.legal_actions[next_player_id] = next_state['legal_actions']
        else:
            actions = [[]for i in range(env.player_num)]
            actions[next_player_id] = next_state['legal_actions']
            new_node = MPMCTSTreeNode(key=action, action=action, legal_actions=actions,parent = node,player_num=env.player_num)
            node.children[action] = new_node

        return new_node


    def caculate_node_UCB(self,node:MPMCTSTreeNode,CP:float,player_id):
        # Q[V] total rewards
        # N[V] visit num
        return node.reward_sum[player_id]/node.visit_num[player_id] + \
               CP*sqrt(2.0*log(node.parent.visit_num[player_id])/node.visit_num[player_id])


    def get_max_UCB_child_node(self,node:MPMCTSTreeNode,CP:float,player_id):
        max_node = None
        max_UCB = -1.0
        for v in node.children.values():
            UCB = self.caculate_node_UCB(v,CP,player_id)
            if UCB > max_UCB:
                max_UCB = UCB
                max_node = v

        return max_node


    # FIND EXPANDABLE NODE
    def tree_policy(self,root_node:MPMCTSTreeNode,env:Env):
        player_id = env.get_player_id()
        untried_action = self.get_untried_action(root_node,player_id)
        if untried_action is not None:
            # 在env上执行action
            node = self.expand_action_on_node(root_node,untried_action,env)
        else:
            # select max-UCB value node
            node = self.get_max_UCB_child_node(root_node,CP_VAL,player_id)
            # step
            next_state,next_player_id = env.step(node.action,False)
            node.legal_actions[next_player_id] = next_state['legal_actions']

        return node

    def default_policy(self, node: MPMCTSTreeNode,env:Env):
        if env.is_over():
            return env.get_payoffs()
        player_id = env.get_player_id()
        #print(player_id)
        action = random.sample(node.legal_actions[player_id], 1)[0]

        while not env.is_over():

            # step forward
            next_state,next_player_id = env.step(action,False)

            if not env.is_over():
                action = random.sample(next_state['legal_actions'],1)[0]
        # game over
        return env.get_payoffs()



    def run_simulation(self,env:Env):
        # while game not over
        # selection, 从tree中往下选择出节点进行扩张(TreePolicy)
        v = self.tree_policy(self.tree_root,env)
        # 扩展后没有结束游戏，进行模拟
        rewards = self.default_policy(v,env)
        while v is not None:
            for i in range(env.player_num):
                v.reward_sum[i] += rewards[i]
                v.visit_num[i] += 1

            v = v.parent

    def save_other_player_cards(self,env:Env):
        cur_player_id = env.get_player_id()
        other_player_id = []
        for i in range(3):
            if i != cur_player_id:
                other_player_id.append(i)

        self.other_hands = [[] for i in range(3)] # 手牌
        self.other_singles = [[] for i in range(3)] # 去重的牌
        for id in other_player_id:
            self.other_hands[id] = copy.deepcopy(env.game.players[id]._current_hand)
            self.other_singles[id] = env.game.players[id].singles



    def restore_other_player_cards(self,env:Env):
        cur_player_id = env.get_player_id()
        other_player_id = []
        for i in range(3):
            if i != cur_player_id:
                other_player_id.append(i)

        for id in other_player_id:
            env.game.players[id]._current_hand = self.other_hands[id]
            env.game.players[id].singles = self.other_singles[id]
            cardstr = cards2str(env.game.players[id]._current_hand)
            env.game.judger.playable_cards[id] = env.game.judger.playable_cards_from_hand(cardstr)


    # 打乱两个人的牌并按照长度重新分配
    def shuffle_other_player_cards(self,env:Env):
        cur_player_id = env.get_player_id()
        other_player_id = []
        for i in range(3):
            if i != cur_player_id:
                other_player_id.append(i)

        other_hands = [[] for i in range(3)]
        # 合并二者
        merged_hands = []
        for id in other_player_id:
            other_hands[id] = copy.deepcopy(env.game.players[id]._current_hand)
            merged_hands.extend(other_hands[id])

        merged_hands = np.asarray(merged_hands)

        other_shuffuled_hands = [[] for i in range(3)]
        other_shuffuled_idx = random.sample(range(len(merged_hands)), len(merged_hands))
        idx_start = 0
        other_shuffled_singels = [[] for i in range(3)]

        for id in other_player_id:
            # 选取
            hands = list(merged_hands[other_shuffuled_idx[idx_start:idx_start + len(other_hands[id])]])
            # 排序
            hands.sort(key=functools.cmp_to_key(doudizhu_sort_card))
            other_shuffuled_hands[id] = hands
            idx_start += len(other_hands[id])

            st = cards2str(other_shuffuled_hands[id])
            cardstr = "".join(OrderedDict.fromkeys(cards2str(other_shuffuled_hands[id])))
            other_shuffled_singels[id] = cardstr

        # 设置数据到env
        for id in other_player_id:
            # 手牌更新
            env.game.players[id]._current_hand = list(other_shuffuled_hands[id])
            # 去重更新
            env.game.players[id].singles = other_shuffled_singels[id]
            # 重新计算可出的牌
            cardstr = cards2str(env.game.players[id]._current_hand)
            env.game.judger.playable_cards[id] = env.game.judger.playable_cards_from_hand(cardstr)






    def step(self,ts):
        # 以当前状态为跟节点进行模拟,
        # 备份env的状态
        # 以当前状态为根节点
        #state = self.env.get_state(player_id=0)  # single player: for blackjack
        legal_actions = ts['legal_actions']
        cur_player_id = self.env.get_player_id() # 真实游戏环境
        legal_actions_list = [[]for i in range(self.env.player_num)]
        legal_actions_list[cur_player_id] = legal_actions

        self.tree_root = MPMCTSTreeNode(key=ROOT_KEY, action=ROOT_ACTION,
                                        legal_actions=legal_actions_list, parent=None,
                                        player_num=self.env.action_num)
        # 模拟次数
        # 备份另外两家的牌
        self.save_other_player_cards(self.env)

        temp_timestep = self.env.timestep

        for i in range(self.emu_num):
            # 洗牌
            self.shuffle_other_player_cards(self.env)
            # 执行模拟
            self.run_simulation(self.env)
            # 恢复初始状态
            while self.env.timestep > temp_timestep:
                self.env.step_back()
                self.env.timestep -= 1

        # 恢复至最初状态
        self.restore_other_player_cards(self.env)
        # 计算每个action的ucb值
        action_ucb = []
        for k,v in self.tree_root.children.items():
            ucb_val = self.caculate_node_UCB(v,0.0,self.env.get_player_id())
            action_ucb.append((self.env._ACTION_LIST[k],ucb_val))
        print(action_ucb)

        # 获取当前玩家最大ucb值
        max_UCB_node = self.get_max_UCB_child_node(self.tree_root,0,self.env.get_player_id())
        cur_hand = self.env.game.players[self.env.get_player_id()].current_hand
        hand_str = ""
        for c in cur_hand:
            if len(c.rank) > 0:
                hand_str += c.rank
            else:
                hand_str += c.suit[0:1]

        print("landlord:{} player_id:{} hand:{} action:{}".format(self.env.game.round.landlord_id,cur_player_id,hand_str,self.env._ACTION_LIST[max_UCB_node.action]))
        return max_UCB_node.action

    def eval_step(self,ts):
        return self.step(ts),self.fake_action_prob



