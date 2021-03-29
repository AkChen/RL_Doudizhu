import tensorflow as tf
import os
import time
import rlcard
from drqn_agent import DRQNAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from rlcard.agents.random_agent import RandomAgent
from SeedRanomAgent import SRandomAgent

# Make environment
from eval_util import general_tournament

train_env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 50
evaluate_num = 200
episode_num = 100000

# The intial memory size
memory_init_size = 100

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/doudizhu_train_drqn_as_L_vs_random_and_eval_vs_random'
loss_log = os.path.join(log_dir,'loss')
L_WR_log = os.path.join(log_dir,'L_WR')
P_WR_log = os.path.join(log_dir,'P_WR')


logger = Logger(log_dir)
loss_logger = Logger(loss_log)
L_WR_logger = Logger(L_WR_log)
P_WR_logger = Logger(P_WR_log)


best_model_path = './models/doudizhu_train_drqn_as_L_vs_random_and_eval_vs_random_best.npy'
max_P_WR = 0.0
max_L_WR = 0.0
# Set a global seed
set_global_seed(0)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DRQNAgent(sess,
                      scope='doudizhu_drqn',
                      action_num=train_env.action_num,
                      memory_init_size=memory_init_size,
                      train_every_t=train_every,
                      state_shape=train_env.state_shape,
                      mlp_layers=[16,32],

                      lstm_units=128
                      )

    train_env.set_agents([agent,RandomAgent(train_env.action_num),RandomAgent(train_env.action_num)])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve


    for episode in range(episode_num):
        print(episode)
        # Generate data from the environment
        agent.reset_step_history()
        trajectories, _ = train_env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        trans_history = []
        for ts in trajectories[0]:
            trans_history.append(ts)
            agent.feed(trans_history)


        #print(episode)
        if episode % evaluate_every == 0:
            eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
            eval_env.set_agents([agent, SRandomAgent(eval_env.action_num, seed=0), SRandomAgent(eval_env.action_num, seed=0)])
            time_start = time.time()
            payoffs1 = general_tournament(eval_env,evaluate_num,False)
            logger.log("episode:{} time:{} landlord winrate:{}".format(episode,time.time()-time_start,payoffs1[0]))
            L_WR_logger.log_performance(episode,payoffs1[0])

            eval_env = rlcard.make('doudizhu', config={'seed': 0, 'allow_step_back': True})
            eval_env.set_agents([SRandomAgent(eval_env.action_num, seed=0), SRandomAgent(eval_env.action_num, seed=0),agent])
            time_start = time.time()
            payoffs2 = general_tournament(eval_env, evaluate_num, False)
            logger.log("episode:{} time:{} peasant winrate:{}".format(episode, time.time() - time_start, payoffs2[1]))
            P_WR_logger.log_performance(episode, payoffs2[1])



            save_flag = False
            if payoffs1[0] > max_L_WR:
                max_L_WR = payoffs1[0]
                save_flag = True
            if payoffs2[1] > max_P_WR:
                max_P_WR = payoffs2[1]
                save_flag = True

            if save_flag:
                print('saving model')
                agent.save_trainable_param_to_file(sess,best_model_path)




    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    loss_logger.plot('DQN loss')
    L_WR_logger.plot('DQN L WR')
    P_WR_logger.plot('DQN P WR')
    # Save model

