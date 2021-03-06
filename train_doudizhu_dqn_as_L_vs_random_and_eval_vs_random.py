import tensorflow as tf
import os
import time
import rlcard
from dqn_agent import DQNAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from rlcard.agents.random_agent import RandomAgent
from SeedRanomAgent import SRandomAgent
import sys

# Make environment
from eval_util import general_tournament

train_env = rlcard.make('doudizhu', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 50
evaluate_num = 100
episode_num = 100000

# The intial memory size
memory_init_size = 100
memory_size = 6000
# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/doudizhu_train_dqn_as_L_vs_random_and_eval_vs_random'
loss_log = os.path.join(log_dir,'loss')
L_WR_log = os.path.join(log_dir,'L_WR')
P_WR_log = os.path.join(log_dir,'P_WR')


logger = Logger(log_dir)
loss_logger = Logger(loss_log)
L_WR_logger = Logger(L_WR_log)
P_WR_logger = Logger(P_WR_log)


best_model_path = './models/doudizhu_train_dqn_as_L_vs_random_and_eval_vs_random_best'
max_P_WR = 0.0
max_L_WR = 0.0
# Set a global seed
set_global_seed(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

with tf.Session(config=config) as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DQNAgent(sess,
                      scope='doudizhu_dqn_L',
                      action_num=train_env.action_num,
                      replay_memory_init_size=memory_init_size,
                        replay_memory_size= memory_size,
                      train_every=train_every,
                      state_shape=train_env.state_shape,
                      mlp_layers=[32,32],
                      )


    train_env.set_agents([agent,RandomAgent(train_env.action_num),RandomAgent(train_env.action_num)])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve


    for episode in range(episode_num):
        print(episode)
        sys.stdout.flush()
        # Generate data from the environment
        trajectories, _ = train_env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent

        for tss in trajectories[:1]:
            for ts in tss:
                agent.feed(ts)



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

            #

            save_flag = False
            if payoffs1[0] > max_L_WR:
                max_L_WR = payoffs1[0]
                save_flag = True
            if payoffs2[1] > max_P_WR:
                max_P_WR = payoffs2[1]
                save_flag = True

            if save_flag:
                print('saving model')
                save_dir = best_model_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(save_dir, 'best_model'))




    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    #loss_logger.plot('DQN loss')
    #L_WR_logger.plot('DQN L WR')
    #P_WR_logger.plot('DQN P WR')
    # Save model
    save_dir = best_model_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'best_model'))

