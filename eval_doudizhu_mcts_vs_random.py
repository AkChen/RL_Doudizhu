import tensorflow as tf
import os
import time
import rlcard
from mcts_doudizhu_agent_ex import MPMCTSAgent
from rlcard.utils import set_global_seed,tournament
from SeedRanomAgent import SRandomAgent
from rlcard.utils import Logger
from eval_util import *



# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_num = 100
emu_num = 25

log_dir = './experiments/doudizhu_mcts_vs_random_result/'

# Set a global seed


# Init a Logger to plot the learning curve
logger = Logger(log_dir)

logger.log("MCTS-UCT VS Random")

# 地主
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0,'allow_step_back':True})
mcts_agent = MPMCTSAgent(eval_env,emu_num=emu_num)
eval_env.set_agents([mcts_agent,SRandomAgent(eval_env.action_num,seed=0),SRandomAgent(eval_env.action_num,seed=0)])

time_start = time.time()
logger.log("MCTS-UCT = landlord winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num,True)[0],time.time()-time_start))



# 农民
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0,'allow_step_back':True})
mcts_agent = MPMCTSAgent(eval_env,emu_num=emu_num)
eval_env.set_agents([SRandomAgent(eval_env.action_num,seed=0),mcts_agent,mcts_agent])

time_start = time.time()
logger.log("MCTS-UCT = peasant  winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num,True)[1],time.time()-time_start))

# Close files in the logger
logger.close_files()



