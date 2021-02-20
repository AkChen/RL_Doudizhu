'''DRQN Agent'''

import tensorflow as tf
import numpy as np


class DRQNAgent(object):

    # 清空rnn历史相关,用于保留历史的state
    def reset_history(self):
        self.total_t = 0 # 重置长度

    def __init__(self,
                 sess,
                 scope,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=2,
                 max_step=50,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 lstm_units=64,
                 learning_rate=0.00005):
        self.sess = sess
        self.scope = scope
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.max_step = max_step
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        # 保存每个时间戳的数据
        self.history_states = [np.random.random(self.state_shape) for e in range(self.max_step)]

        # Total timesteps
        self.total_t = 0


        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        # Create estimators
        self.q_estimator = Estimator(scope=self.scope + "_q", action_num=action_num,
                                     learning_rate=learning_rate,max_step=max_step,
                                     state_shape=state_shape, mlp_layers=mlp_layers,
                                     lstm_units=lstm_units)

        self.target_estimator = Estimator(scope=self.scope + "_target_q", action_num=action_num,
                                          learning_rate=learning_rate, max_step=max_step,
                                          state_shape=state_shape,mlp_layers=mlp_layers,
                                          lstm_units=lstm_units)

    # return data:[1,max_step,state_shape],step_length:[1]

    # 训练的时候进行step
    def step(self,state):
        # 放入数据
        self.history_states[self.total_t] = state['obs']
        self.total_t += 1
        if(self.total_t>self.max_step):
            print("totoal_t is lagger than max_step")
        estimator_data,step_length = np.asarray([self.history_states]),np.asarray([self.total_t])
        action_prob = self.predict(estimator_data,step_length)[0] # batch_size =1 取第一个
        action_prob = remove_illegal(action_prob, state['legal_actions'])
        # 根据概率选择
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return action

    # 评估的时候进行step,直接通过estimator评估
    def eval_step(self,state):
        self.history_states[self.total_t] = state['obs']
        self.total_t += 1
        if (self.total_t > self.max_step):
            print("totoal_t is lagger than max_step")
        estimator_data, step_length = np.asarray([self.history_states]), np.asarray([self.total_t])
        q_values = self.q_estimator.predict(self.sess, estimator_data, step_length)[0] # batch_size = 1 所以取第一个
        probs = remove_illegal(np.exp(q_values), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    def predict(self, batch_state_seq,batch_length):
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A = []
        for e in batch_length:
            row = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
            A.append(row)
        A = np.asarray(A)
        q_values = self.q_estimator.predict(self.sess,batch_state_seq,batch_length) # [batch_size,action_num]
        best_actions = np.argmax(q_values,axis=1) # 取第二维,[batch_size]
        for i,action in enumerate(best_actions):
            A[i][action] += (1.0 - epsilon)
        return A



class Estimator():
    def __init__(self, scope="estimator", action_num=2, learning_rate=0.001, max_step=50, state_shape=None,
                 mlp_layers=None, lstm_units=10):

        # 网络架构[ input->flatten->mlp_layers[0...N-1]->lstm_layers[0...M-1]->action_num]
        self.scope = scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.state_shape = state_shape if isinstance(state_shape, list) else [state_shape]
        self.mlp_layers = map(int, mlp_layers)
        self.lstm_units = lstm_units
        self.max_time_step = max_step

        with tf.variable_scope(scope):
            self._build_model()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='dqn_adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def _build_model(self):
        # 时间长度 每个样本的时间长度
        self.T_pl = tf.placeholder(shape=[None], dtype=tf.int32, name='T')
        # 构建输入形状
        # 由于是lstm所以应该是[batch_size,time_step,state_shape[...]]
        input_shape = [None, self.max_time_step]  # 第一维为batch,第二维为时间戳
        input_shape.extend(self.state_shape)
        self.X_pl = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        # 注意[batch_size,time_step]
        self.y_pl = tf.placeholder(shape=[None, self.max_time_step], dtype=tf.float32, name="y")

        self.actions_pl = tf.placeholder(shape=[None, self.max_time_step], dtype=tf.int32, name="actions")

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        state_flatten_dim = 1
        for s in self.state_shape:
            state_flatten_dim *= s

        X = tf.layers.batch_normalization(self.X_pl, training=self.is_train)

        fc = tf.reshape(X, [-1, self.max_time_step, state_flatten_dim])

        for dim in self.mlp_layers:
            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.tanh)

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units)
        batch_size = tf.shape(self.X_pl)[0]
        self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
        self.lstm_output, final_states = tf.nn.dynamic_rnn(cell=self.cell, inputs=fc, initial_state=self.init_state,dtype=tf.float32, sequence_length=self.T_pl)

        self.predictions = tf.contrib.layers.fully_connected(self.lstm_output, self.action_num, activation_fn=None)
        # [batch_size,max_step,action_num]

        # 计算loss

        gather_indices = tf.range(batch_size * self.max_time_step) * self.action_num + tf.reshape(self.actions_pl, [-1])
        self.action_predictions = tf.reshape(tf.gather(tf.reshape(self.predictions, [-1]), gather_indices),
                                             [batch_size, self.max_time_step])

        self.losses = tf.reduce_sum(tf.square(self.y_pl - self.action_predictions), axis=1)
        self.loss = tf.reduce_mean(self.losses)

        # 预测相关


    def predict(self,sess,batch_state_seq,batch_length):
        prediction_prob = sess.run(self.predictions,feed_dict = {self.X_pl:batch_state_seq,self.T_pl:batch_length})
        # batch_length[...] >= 1
        # 取出最后一个时间戳的结果
        prediction_final_step_prob = []
        for i,prob in enumerate(prediction_prob):
            prediction_final_step_prob.append(prob[i][batch_length[i]-1])
        return np.asarray(prediction_final_step_prob) # [batch_size,action_num]




def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs


