import tensorflow as tf
import os
import time
import rlcard

from rlcard.utils import set_global_seed, tournament
from SeedRanomAgent import SRandomAgent
from rlcard.utils import Logger
from eval_util import *
from mcts_drqn_doudizhu_agent import MPMCTSDRQNAgent
from mcts_doudizhu_agent_ex import MPMCTSAgent
from dqn_agent import DQNAgent

# load dqn
best_model_path = './models/doudizhu_train_drqn_as_L_vs_random_and_eval_vs_random_best.npy'
env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


sess = tf.Session(config=config)

drqn_agent = DRQNAgent(sess,
                      scope='doudizhu_drqn',
                      action_num=env.action_num,
                      memory_init_size=3000,
                      memory_size=6000,
                      train_every_t=1,
                      state_shape=env.state_shape,
                       mlp_layers=[32], lstm_units=32)

sess.run(tf.global_variables_initializer())
drqn_agent.load_trainable_param_from_file(sess,best_model_path)

drqn_agent_copy_1 = DRQNAgent(sess,
                      scope='doudizhu_drqn_cp1',
                      action_num=env.action_num,
                      memory_init_size=3000,
                      memory_size=6000,
                      train_every_t=1,
                      state_shape=env.state_shape,
                      mlp_layers=[32],lstm_units=32,param_dict=drqn_agent.get_param_dict())
drqn_agent_copy_2 = DRQNAgent(sess,
                      scope='doudizhu_drqn_cp2',
                      action_num=env.action_num,
                      memory_init_size=3000,
                      memory_size=6000,
                      train_every_t=1,
                      state_shape=env.state_shape,
                              mlp_layers=[32], lstm_units=32,param_dict=drqn_agent.get_param_dict())


# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_num = 100
emu_num = 50

log_dir = './experiments/doudizhu_mcts_drqn_vs_mcts_result/'

# Set a global seed

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

logger.log("MCTS-UCT+DRQN VS MCTS")

# 地主
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_drqn_agent = MPMCTSDRQNAgent(eval_env, emu_num=emu_num, drqn_agents=[drqn_agent, drqn_agent_copy_1, drqn_agent_copy_2])
mcts_agent = MPMCTSAgent(eval_env,emu_num=emu_num)
eval_env.set_agents([mcts_drqn_agent, mcts_agent, mcts_agent])

time_start = time.time()
logger.log("MCTS-UCT+DRQN = landlord winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[0],
                                                               time.time() - time_start))

# 农民
set_global_seed(0)
eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
mcts_drqn_agent = MPMCTSDRQNAgent(eval_env, emu_num=emu_num, drqn_agents=[drqn_agent, drqn_agent_copy_1, drqn_agent_copy_2])
mcts_agent = MPMCTSAgent(eval_env,emu_num=emu_num)
eval_env.set_agents([mcts_agent,mcts_drqn_agent,mcts_drqn_agent])

time_start = time.time()
logger.log("MCTS-UCT+DRQN = peasant  winrate:{} time:{}".format(general_tournament(eval_env, evaluate_num, True)[1],
                                                               time.time() - time_start))

# Close files in the logger
logger.close_files()
