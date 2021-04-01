import tensorflow as tf
import os

import rlcard
from mcts_doudizhu_agent_ex import MPMCTSAgent
from rlcard.utils import set_global_seed,tournament
from rlcard.agents import RandomAgent
from rlcard.utils import Logger
from eval_util import general_tournament



eval_env = rlcard.make('doudizhu', config={'seed': 0,'allow_step_back':True})


# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 1000
evaluate_num = 10000
episode_num = 100000

log_dir = './experiments/blackjack_mcts_result/'

# Set a global seed
set_global_seed(0)

# Set up the agents
agent = MPMCTSAgent(eval_env,emu_num=100)
rdm_agent = RandomAgent(action_num=eval_env.action_num)
eval_env.set_agents([agent,rdm_agent,rdm_agent])

eval_env.run(is_training=False)
print(eval_env.game.round.trace)
# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):
    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(eval_env.timestep,general_tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('MCTS')

# Save model
save_dir = 'models/blackjack_mcts'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saver = tf.train.Saver()

