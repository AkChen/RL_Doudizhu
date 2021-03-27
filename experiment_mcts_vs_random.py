import tensorflow as tf
import os

import rlcard
from mcts_doudizhu_agent_ex import MPMCTSAgent,mcts_tournament
from rlcard.utils import set_global_seed,tournament
from SeedRanomAgent import SRandomAgent
from rlcard.utils import Logger



eval_env_1 = rlcard.make('doudizhu', config={'seed': 0,'allow_step_back':True})
eval_env_2 = rlcard.make('doudizhu', config={'seed': 0,'allow_step_back':True})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_num = 100
emu_num = 100

log_dir = './experiments/doudizhu_mcts_vs_random_result/'

# Set a global seed
set_global_seed(0)

# Set up the agents
agent = MPMCTSAgent(eval_env_1,emu_num=emu_num)
rdm_agent = SRandomAgent(action_num=eval_env_1.action_num,seed=0)
eval_env_1.set_agents([agent,rdm_agent,rdm_agent]) # mcts当地主
agent = MPMCTSAgent(eval_env_2,emu_num=emu_num)
rdm_agent = SRandomAgent(action_num=eval_env_1.action_num,seed=0)
eval_env_2.set_agents([rdm_agent,agent,agent]) # mcts当农民

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

logger.log("MCTS-UCT VS Random")
logger.log("MCTS-UCT = 地主")
logger.log_performance(eval_env_1.timestep,mcts_tournament(eval_env_1, evaluate_num)[0])
logger.log("MCTS-UCT = 农民")
logger.log_performance(eval_env_2.timestep,mcts_tournament(eval_env_2, evaluate_num)[1])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('MCTS')

# Save model
save_dir = 'models/blackjack_mcts'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saver = tf.train.Saver()

