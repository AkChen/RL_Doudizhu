import tensorflow as tf
import os
import time
import rlcard

from rlcard.utils import set_global_seed, tournament
from SeedRanomAgent import SRandomAgent
from rlcard.utils import Logger
from eval_util import *
from mcts_doudizhu_agent_ex import MPMCTSAgent
from dqn_agent import DQNAgent

# load dqn
best_model_path = './models/doudizhu_train_dqn_as_L_vs_random_and_eval_vs_random_best'
e = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


sess = tf.Session(config=config)
dqn_agent = DQNAgent(sess,
                     scope='doudizhu_dqn_L',
                     action_num=e.action_num,
                     replay_memory_init_size=1,
                     train_every=1,
                     state_shape=e.state_shape,
                     mlp_layers=[32, 32])

sess.run(tf.global_variables_initializer())
save_dir = os.path.join(best_model_path, 'best_model')
saver = tf.train.Saver()
saver.restore(sess, save_dir)

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_num = 100
emu_num = 50

log_dir = './experiments/doudizhu_dqn_vs_mcts_result/'

# Set a global seed


# Init a Logger to plot the learning curve
logger = Logger(log_dir)

logger.log("DQN VS MCTS")

# 地主
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_agent = MPMCTSAgent(eval_env, emu_num=emu_num)
eval_env.set_agents([dqn_agent,mcts_agent, mcts_agent])

time_start = time.time()
logger.log("DQN = landlord winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[0],
                                                               time.time() - time_start))

# 农民
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_agent = MPMCTSAgent(eval_env, emu_num=emu_num)
eval_env.set_agents([mcts_agent, dqn_agent, dqn_agent])

time_start = time.time()
logger.log("DQN= peasant  winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[1],
                                                               time.time() - time_start))

# Close files in the logger
logger.close_files()
sess.close()