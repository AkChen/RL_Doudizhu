from drqn_agent import DRQNAgent
import rlcard
import tensorflow as tf

env = rlcard.make('doudizhu')

sess = tf.Session()

agent1 = DRQNAgent(sess,'drqn1',max_step=40,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)
agent2 = DRQNAgent(sess,'drqn2',max_step=40,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)
agent3 = DRQNAgent(sess,'drqn3',max_step=40,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)

sess.run(tf.global_variables_initializer())


env.set_agents([agent1,agent2,agent3])

trans,_ = env.run(is_training=True)
print(len(trans[0]))