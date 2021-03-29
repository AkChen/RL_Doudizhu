# 评测方法
from drqn_agent import DRQNAgent
from SeedRanomAgent import SRandomAgent
from dqn_agent import DQNAgent
from mcts_doudizhu_agent_ex import MPMCTSAgent
import rlcard

def general_tournament(env,num,verbose = False):
    payoffs = [0 for _ in range(env.player_num)]
    counter = 0
    while counter < num:
        for agent in env.agents:
            if agent.__class__.__name__ == 'DRQNAgent':
                agent.reset_step_history()

        _, _payoffs = env.run(is_training=False)
        if verbose:
            print("counter:{}/{},result:{}".format(counter + 1, num,_payoffs))

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


# getter() return agent
# 需要设置随机种子的agent外部定制

class AgentGetter(object):

    def get_agent(self):
        pass



class DRQNAgentGetter(AgentGetter):

    def __init__(self,agent:DRQNAgent):
        self.agent = agent

    def get_agent(self):
        return self.agent


class SRandomAgentGetter(AgentGetter):
    def __init__(self, action_num, seed):
        self.action_num = action_num
        self.seed = seed

    def get_agent(self):
        return SRandomAgent(self.action_num, self.seed)

class DQNAgentGetter(AgentGetter):

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def get_agent(self):
        return self.agent

class MPMCTSGetter(AgentGetter):
    def __init__(self, agent: MPMCTSAgent):
        self.agent = agent

    def get_agent(self):
        return self.agent


