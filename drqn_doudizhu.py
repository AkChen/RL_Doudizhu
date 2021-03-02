import tensorflow as tf
import os

import rlcard
from rlcard import models
from drqn_agent import DRQNAgent, drqn_tournament
from rlcard.agents import CFRAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('doudizhu', config={'seed': 0})
eval_env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
evaluate_num = 500
episode_num = 100000

# The intial memory size
memory_init_size = 100
# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves

log_dirs = ['./experiments/doudizhu_drqn_result_1/','./experiments/doudizhu_drqn_result_2/','./experiments/doudizhu_drqn_result_3/']
# Set a global seed
set_global_seed(0)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent_1 = DRQNAgent(sess,
                      scope='drqn_1',
                      action_num=env.action_num,
                      memory_init_size=memory_init_size,
                      train_every_t=train_every,
                      state_shape=env.state_shape,
                      mlp_layers=[64,64],
                      lstm_units=128)
    agent_2 = DRQNAgent(sess,
                        scope='drqn_2',
                        action_num=env.action_num,
                        memory_init_size=memory_init_size,
                        train_every_t=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        lstm_units=128)
    agent_3 = DRQNAgent(sess,
                        scope='drqn_3',
                        action_num=env.action_num,
                        memory_init_size=memory_init_size,
                        train_every_t=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        lstm_units=128)
    agents = [agent_1,agent_2,agent_3]

    env.set_agents(agents)
    eval_env.set_agents(agents)

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve


    loggers = [Logger(log_dirs[i]) for i in range(3)]

    for episode in range(episode_num):

        # Generate data from the environment
        agent_1.reset_step_history()
        agent_2.reset_step_history()
        agent_3.reset_step_history()
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i,trans in enumerate(trajectories):
            trans_history = []
            for ts in trans:
                trans_history.append(ts)
                agents[i].feed(trans_history)

        #print(episode)
        if episode % evaluate_every == 0:
            payoff = drqn_tournament(eval_env, evaluate_num)
            for i in range(3):
                loggers[i].log_performance(env.timestep, payoff[i])

    # Close files in the logger
    for i in range(3):
        loggers[i].close_files()

    # Plot the learning curve
    for i in range(3):
        loggers[i].plot('DRQN')

    # Save model
    save_dir = 'models/doudizhu_drqn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
