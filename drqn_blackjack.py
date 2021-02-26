from drqn_agent import DRQNAgent,drqn_tournament
import rlcard
from rlcard.utils import tournament
from dqn_agent import DQNAgent
import tensorflow as tf
from rlcard.utils import Logger

env = rlcard.make('blackjack')
eval_env = rlcard.make('blackjack')

sess = tf.Session()
max_step = 20
#agent = DRQNAgent(sess,'drqn1',max_step=max_step,mlp_layers=[32,64],lstm_units=64,
                  #state_shape=env.state_shape,action_num=env.action_num,memory_size=100,show_info_every=200)
agent = DQNAgent(sess,'dqn',state_shape=env.state_shape,action_num=env.action_num,mlp_layers=[10,10],replay_memory_init_size=100)


sess.run(tf.global_variables_initializer())


env.set_agents([agent])
eval_env.set_agents([agent])


log_dir = './experiments/blackjack_drqn_result/'
logger = Logger(log_dir)

X_round = 100000000 # rouds of game
for i in range(X_round):
    #agent.reset_step_history()
    all_trans,_ = env.run(is_training=True)
    agent_trans = all_trans[0]
    trans_list = []
    for j,trans in enumerate(agent_trans):
            trans_list.append(trans)
            agent.feed(trans)
    #print("epoch:{}\n".format(i))
    if i % 3000 == 0:
        logger.log_performance(env.timestep, tournament(eval_env, 1000)[0])



