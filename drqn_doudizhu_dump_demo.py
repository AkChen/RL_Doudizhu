import tensorflow as tf
import os

import rlcard
from rlcard.utils import set_global_seed
from drqn_agent import DRQNAgent


env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
evaluate_num = 500
episode_num = 1

# The intial memory size
memory_init_size = 100
# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves

log_dirs = ['./experiments/doudizhu_drqn_result_1/','./experiments/doudizhu_drqn_result_2/','./experiments/doudizhu_drqn_result_3/']
# Set a global seed
set_global_seed(0)
model_file_name = "./models/doudizhu_dump_demp_drqn.npy"
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

    param_dict = agent_1.get_param_dict()

    agent_2 = DRQNAgent(sess,
                        scope='drqn_2',
                        action_num=env.action_num,
                        memory_init_size=memory_init_size,
                        train_every_t=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        lstm_units=128,param_dict=param_dict)


    agent_3 = DRQNAgent(sess,
                        scope='drqn_3',
                        action_num=env.action_num,
                        memory_init_size=memory_init_size,
                        train_every_t=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[64, 64],
                        lstm_units=128,param_dict=param_dict)
    train_agents = [agent_1,agent_2,agent_3]

    env.set_agents(train_agents)

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    agent_1.save_trainable_param_to_file(sess, model_file_name)

    agent_1.load_trainable_param_from_file(sess,model_file_name)


