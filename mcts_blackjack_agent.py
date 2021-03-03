import copy
from rlcard.envs import Env
import time
import random
from math import log,sqrt
# single player tree
ROOT_KEY = 0
ROOT_ACTION = 0
CP_VAL = 0.70710678118655

class MCTSTreeNode(object):

    def __init__(self,key,action,legal_actions: list,parent = None,game_over = False):
        self.key = key
        self.action = action
        self.legal_actions = legal_actions
        self.parent = parent
        self.visit_num = 1  # set at least once, convenient for divide opt.
        self.reward_sum = 0.0
        self.game_over = game_over
        self.children = dict() # key:action val: MCTSTreeNode TODO:: multi-player impl key:(player_id,action) tuple

        # constant




class MCTSAgent(object):

    def __init__(self,env:Env, # for simulation
):
        self.env = env
        self.use_raw = False
        self.fake_action_prob = [0.0 for i in range(env.action_num)]

    # 每次run之前都要初始化env，与外部保持一致


    def get_untried_action(self,node:MCTSTreeNode):
        for legal_action in node.legal_actions:
            if not node.children.__contains__(legal_action):
                return legal_action
        return None


    # return expanded child node
    def expand_action_on_node(self,node:MCTSTreeNode,action,env:Env):
        # 进行step，获得进行这次action的legal_actions
        next_state,next_player_id = env.step(action,False)
        new_node = MCTSTreeNode(key=action,action=action,legal_actions=next_state['legal_actions'],
                                game_over=env.is_over(),parent=node)
        node.children[action] = new_node

        return new_node


    def caculate_node_UCB(self,node:MCTSTreeNode,CP:float):
        # Q[V] total rewards
        # N[V] visit num
        return node.reward_sum/node.visit_num + CP*sqrt(2.0*log(node.parent.visit_num)/node.visit_num)

    def get_max_UCB_child_node(self,node:MCTSTreeNode,CP:float):
        max_node = None
        max_UCB = -1.0
        for v in node.children.values():
            UCB = self.caculate_node_UCB(v,CP)
            if UCB > max_UCB:
                max_UCB = UCB
                max_node = v

        return max_node


    # FIND EXPANDABLE NODE
    def tree_policy(self,root_node:MCTSTreeNode,env:Env):
        untried_action = self.get_untried_action(root_node)
        if untried_action is not None:
            # 在env上执行action
            node = self.expand_action_on_node(root_node,untried_action,env)
        else:
            # select max-UCB value node
            node = self.get_max_UCB_child_node(root_node,CP_VAL)
            # step
            next_state,next_player_id = env.step(node.action,False)
            node.legal_actions = next_state['legal_actions']
            node.game_over = env.is_over()

        return node

    def default_policy(self, node: MCTSTreeNode,env:Env):
        action = random.sample(node.legal_actions, 1)[0]

        while not env.is_over():
            # step forward
            next_state,next_player_id = env.step(action,False)
            if not env.is_over():
                action = random.sample( next_state['legal_actions'],1)[0]
        # game over
        reward = env.get_payoffs()[0]
        return reward



    def run_simulation(self,env:Env):
        # while game not over
        # selection, 从tree中往下选择出节点进行扩张(TreePolicy)
        # 洗牌，这样才能保证后面的牌是随机的（相对于真正的发牌顺序）
        env.game.dealer.shuffle()
        v = self.tree_policy(self.tree_root,env)
        reward = self.default_policy(v,env)
        while v is not None:
            v.reward_sum += reward
            v.visit_num +=1
            v = v.parent

    def step(self,ts):
        # 以当前状态为跟节点进行模拟,
        # 备份env的状态
        # 以当前状态为根节点
        #state = self.env.get_state(player_id=0)  # single player: for blackjack
        legal_actions = ts['legal_actions']#state['legal_actions']
        self.tree_root = MCTSTreeNode(key=ROOT_KEY, action=ROOT_ACTION, legal_actions=legal_actions, parent=None)
        # 模拟次数
        temp_timestep = self.env.timestep
        for i in range(100):
            #env_copy = copy.deepcopy(self.env)
            self.run_simulation(self.env)
            while self.env.timestep > temp_timestep:
                self.env.timestep -= 1
                self.env.step_back()

        max_UCB_node = self.get_max_UCB_child_node(self.tree_root,0)

        return max_UCB_node.action

    def eval_step(self,ts):
        return self.step(ts),self.fake_action_prob


def mcts_tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.player_num)]
    counter = 0
    while counter < num:
        print(counter)
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs
