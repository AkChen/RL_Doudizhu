from rlcard.agents import RandomAgent
import rlcard
import numpy  as np
env = rlcard.make('doudizhu')

env.set_agents([RandomAgent(env.action_num),RandomAgent(env.action_num),RandomAgent(env.action_num)])

# 让他们进行一轮斗地主

a = 0
for i in range(1000):
    trans,_ = env.run(is_training=True)
    if(len(trans[0]) > a):
        print(len(trans[0]))
        a = len(trans[0])


print('')