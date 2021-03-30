import tensorflow as tf
import os
import time
import rlcard

from rlcard.utils import set_global_seed, tournament
from SeedRanomAgent import SRandomAgent
from rlcard.utils import Logger
from eval_util import *
from mcts_dqn_doudizhu_agent import MPMCTSDQNAgent
from dqn_agent import DQNAgent

# load dqn
best_model_path = './models/doudizhu_train_dqn_vs_dqn_and_eval_vs_random_best'
e = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
sess = tf.Session()
dqn_agent = DQNAgent(sess,
                     scope='doudizhu_dqn',
                     action_num=e.action_num,
                     replay_memory_init_size=1,
                     train_every=1,
                     state_shape=e.state_shape,
                     mlp_layers=[16, 32])

sess.run(tf.global_variables_initializer())
save_dir = os.path.join(best_model_path, 'best_model')
saver = tf.train.Saver()
saver.restore(sess, save_dir)

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_num = 100
emu_num = 10

log_dir = './experiments/doudizhu_mcts_dqn_vs_random_result/'

# Set a global seed


# Init a Logger to plot the learning curve
logger = Logger(log_dir)

logger.log("MCTS-UCT+DQN VS Random")

# 地主
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_agent = MPMCTSDQNAgent(eval_env, emu_num=emu_num, dqn_agents=[dqn_agent, dqn_agent, dqn_agent])
eval_env.set_agents([mcts_agent, SRandomAgent(eval_env.action_num, seed=0), SRandomAgent(eval_env.action_num, seed=0)])

time_start = time.time()
logger.log("MCTS-UCT+DQN = landlord winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[0],
                                                               time.time() - time_start))

# 农民
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_agent = MPMCTSDQNAgent(eval_env, emu_num=emu_num, dqn_agents=[dqn_agent, dqn_agent, dqn_agent])
eval_env.set_agents([SRandomAgent(eval_env.action_num, seed=0), SRandomAgent(eval_env.action_num, seed=0), mcts_agent])

time_start = time.time()
logger.log("MCTS-UCT+DQN = peasant  winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[1],
                                                               time.time() - time_start))

# Close files in the logger
logger.close_files()