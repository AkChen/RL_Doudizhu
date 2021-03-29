import tensorflow as tf
import os

import rlcard
from drqn_agent import DRQNAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
from eval_util import general_tournament

env = rlcard.make('blackjack', config={'seed': 0})
eval_env = rlcard.make('blackjack', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 1000
evaluate_num = 10000
episode_num = 100000

# The intial memory size
memory_init_size = 100

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/blackjack_drqn_result/'

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DRQNAgent(sess,
                      scope='drqn',
                      action_num=env.action_num,
                      memory_init_size=memory_init_size,
                      train_every_t=train_every,
                      state_shape=env.state_shape,
                      mlp_layers=[16,32],
                      lstm_units=32)

    env.set_agents([agent])
    eval_env.set_agents([agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    for episode in range(episode_num):

        # Generate data from the environment
        agent.reset_step_history()
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        trans_history = []
        for ts in trajectories[0]:
            trans_history.append(ts)
            agent.feed(trans_history)

        #print(episode)
        if episode % evaluate_every == 0:
            print('')
            logger.log_performance(env.timestep, general_tournament(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DRQN')

    # Save model
    save_dir = 'models/blackjack_drqn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
