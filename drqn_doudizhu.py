from drqn_agent import DRQNAgent
import rlcard
import tensorflow as tf

env = rlcard.make('doudizhu')

sess = tf.Session()
max_step = 10
agent1 = DRQNAgent(sess,'drqn1',max_step=max_step,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)
agent2 = DRQNAgent(sess,'drqn2',max_step=max_step,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)
agent3 = DRQNAgent(sess,'drqn3',max_step=max_step,mlp_layers=[32,64],lstm_units=64,
                  state_shape=env.state_shape,action_num=env.action_num)



sess.run(tf.global_variables_initializer())

agents = [agent1,agent2,agent3]
env.set_agents(agents)

X_round = 100000 # rouds of game
for i in range(X_round):
    for agent in agents:
        agent.reset_step_history()
    all_trans,_ = env.run(is_training=True)
    for agent_i,agent in enumerate(agents):
        agent_trans = all_trans[agent_i]
        trans_list = []
        for j,trans in enumerate(agent_trans):
            trans_list.append(trans)
            agent.feed(trans_list)
        #agent.feed(agent_trans)

