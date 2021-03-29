'''DRQN Agent implemented by: Crystal Yin & AkChen'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops

class DRQNAgent(object):

    def reset_train_history(self):
        self.cur_t = 0
        if self.total_t < self.memory_size:
            high_bound = self.total_t
        else:
            high_bound = self.memory_size
        for i in range(high_bound):
            self.memory_history_steps[i] = 0

    # 每次run之前需要调用
    def reset_step_history(self):
        self.cur_step = 0

    def __init__(self,
                 sess,
                 scope,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,  # 每次train传入batch_size个数据，而step传入一个数据
                 memory_size=20000,
                 memory_init_size=100,
                 max_step=50,
                 action_num=2,
                 state_shape=None,
                 train_every_t=1,  # train drqn every trans
                 show_info_every=50,
                 mlp_layers=None,
                 lstm_units=64,
                 learning_rate=0.00005,
                 param_dict=None):
        self.use_raw = False
        self.sess = sess
        self.scope = scope
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_init_size = memory_init_size
        assert (self.batch_size <= self.memory_init_size)
        assert (self.memory_init_size <= self.memory_size)
        self.action_num = action_num
        self.train_every_t = train_every_t
        self.max_step = max_step
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        # for training temp
        # [batch_size,max_step,state_shape]
        temp_shape = [self.memory_size, max_step]
        for e in self.state_shape:
            temp_shape.append(e)
        self.memory_history_states = np.zeros(temp_shape).astype(float)
        self.memory_history_next_states = np.zeros(temp_shape).astype(float)
        temp_shape = [self.memory_size, max_step]
        self.memory_history_actions = np.zeros(temp_shape).astype(int)
        self.memory_history_reward = np.zeros(temp_shape).astype(float)
        self.memory_history_done = np.ones(temp_shape).astype(bool)
        self.memory_history_steps = np.zeros([self.memory_size]).astype(int)

        # for step temp
        temp_shape = [max_step]
        for e in self.state_shape:
            temp_shape.append(e)
        self.history_states = np.zeros(temp_shape).astype(float)

        # Total time step,(fed into memory)
        self.total_t = 0
        # cur time step position, (fed into memory)
        self.cur_t = 0
        # cur step position
        self.cur_step = 0
        # train num
        self.train_t = 0

        self.show_info_every = show_info_every

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        # Create estimators
        if param_dict is None:
            param_dict = dict()
            param_dict['q_net'] = None
            param_dict['target_q_net'] = None

        self.q_estimator = Estimator(scope=self.scope + "_q", action_num=action_num,
                                     learning_rate=learning_rate, max_step=max_step,
                                     state_shape=state_shape, mlp_layers=mlp_layers,
                                     lstm_units=lstm_units, param_dict=param_dict['q_net'])

        self.target_estimator = Estimator(scope=self.scope + "_target_q", action_num=action_num,
                                          learning_rate=learning_rate, max_step=max_step,
                                          state_shape=state_shape, mlp_layers=mlp_layers,
                                          lstm_units=lstm_units, param_dict=param_dict['target_q_net'])

    # for training
    # feed a time-step sample into memory
    def feed(self, trans):
        # randomly select a tans that length >= 1
        assert len(trans) >= 1

        random_start_index = 0  # np.random.randint(low=0,high=len(trans))
        random_start_trans = trans[random_start_index:]
        for t, ts in enumerate(random_start_trans):
            assert (t < self.max_step)
            (state, action, reward, next_state, done) = tuple(ts)
            self.memory_history_states[self.cur_t][t] = state['obs']
            self.memory_history_actions[self.cur_t][t] = action
            self.memory_history_reward[self.cur_t][t] = reward
            self.memory_history_next_states[self.cur_t][t] = next_state['obs']
            self.memory_history_done[self.cur_t][t] = done

        # set length
        self.memory_history_steps[self.cur_t] = len(random_start_trans)
        self.total_t = self.total_t + 1
        # cycle feeding
        self.cur_t = self.total_t % self.memory_size

        # need to train ?
        if self.total_t >= self.memory_init_size and (self.total_t - self.memory_init_size) % self.train_every_t == 0:
            self.batch_train()
            # self.reset_train_history()

        # make sure the source q-net have been trained as least once.
        if self.total_t >= self.memory_init_size and self.train_t % self.update_target_estimator_every == 0:
            self.copy_esimator_params(self.sess, self.q_estimator, self.target_estimator)
            #print("INFO - copy params finished\n")

    def copy_esimator_params(self, sess, estimator1, estimator2):

        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        sess.run(update_ops)
        return 0

    # return estimator_data, step_length
    def sample_from_memory(self):
        if self.total_t < self.memory_size:
            high_bound = self.total_t
        else:
            high_bound = self.memory_size

        # assert(high_bound >= self.batch_size)

        random_index = np.random.randint(low=0, high=high_bound, size=self.batch_size)
        batch_step_length = self.memory_history_steps[random_index]
        batch_states = self.memory_history_states[random_index]
        batch_next_states = self.memory_history_next_states[random_index]
        batch_actions = self.memory_history_actions[random_index, batch_step_length - 1]
        batch_reward = self.memory_history_reward[random_index, batch_step_length - 1]
        batch_done = self.memory_history_done[random_index, batch_step_length - 1]

        return batch_states, batch_actions, batch_reward, batch_next_states, batch_done, batch_step_length

    # 批量训练
    def batch_train(self):
        # random sample from memory
        batch_states, batch_actions, batch_reward, batch_next_states, batch_done, batch_step_length = self.sample_from_memory()

        # 通过q_net预测batch中每个样本在每个action上的概率
        q_value_next = self.q_estimator.predict(self.sess, batch_next_states, batch_step_length)
        best_next_actions = np.argmax(q_value_next, axis=1)
        q_values_next_target = self.target_estimator.predict(self.sess, batch_next_states, batch_step_length)

        target_batch = batch_reward + np.invert(batch_done).astype(np.float32) * \
                       self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_next_actions]

        loss = self.q_estimator.update(self.sess, batch_states, batch_actions, target_batch, batch_step_length)
        self.train_t += 1
        #if self.train_t % self.show_info_every == 0:
            #print('INFO - Agent {}, step {}, rl-loss: {}\n'.format(self.scope, self.total_t, loss), end='')

        return loss

    # 训练的时候进行step，同时保存数据到历史中
    def step(self, state):
        self.history_states[self.cur_step] = state['obs']
        # 历史时长增加
        self.cur_step += 1

        #if self.cur_step > self.max_step:
            #print("step:cur_step is lagger than max_step")

        estimator_data, step_length = np.asarray([self.history_states]), np.asarray([self.cur_step])
        action_prob = self.predict(estimator_data, step_length)[0]  # batch_size =1 取第一个
        action_prob = remove_illegal(action_prob, state['legal_actions'])
        # 根据概率选择
        action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        return action

    # 评估的时候进行step,直接通过estimator评估
    def eval_step(self, state):
        self.history_states[self.cur_step] = state['obs']
        self.cur_step += 1
        if self.cur_step > self.max_step:
            print("eval_step:cur_step is lagger than max_step")

        estimator_data, step_length = np.asarray([self.history_states]), np.asarray([self.cur_step])
        q_values = self.q_estimator.predict(self.sess, estimator_data, step_length)[0]  # batch_size = 1 所以取第一个
        probs = remove_illegal(np.exp(q_values), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    # for train
    def predict(self, batch_state_seq, batch_length):
        # batch_state_seq : [batch_size,max_step,data_length]
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A = []
        for e in batch_length:
            row = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
            A.append(row)
        A = np.asarray(A)
        q_values = self.q_estimator.predict(self.sess, batch_state_seq, batch_length)  # [batch_size,action_num]
        best_actions = np.argmax(q_values, axis=1)  # 取第二维,[batch_size]
        for i, action in enumerate(best_actions):
            A[i][action] += (1.0 - epsilon)
        return A

    def get_param_dict(self):
        dic = dict()
        dic['q_net'] = self.q_estimator.get_param_dict()
        dic['target_q_net'] = self.target_estimator.get_param_dict()
        return dic

    # 保存可训练参数至文件

    def save_trainable_param_to_file(self,sess,file_name):
        # 每个estimator的可训练参数
        #param_dict = self.get_param_dict()

        q_net_dict = dict()
        target_net_dict = dict()

        q_net_params = [t for t in tf.trainable_variables() if t.name.startswith(self.q_estimator.scope)]
        q_net_params = sorted(q_net_params, key=lambda v: v.name)
        for p in q_net_params:
            name = p.name
            #$print(name)
            q_net_dict[name.strip(self.q_estimator.scope)] = sess.run(p)

        target_net_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target_estimator.scope)]
        target_net_params = sorted(target_net_params, key=lambda v: v.name)
        for p in target_net_params:
            name = p.name
            #print(name)
            target_net_dict[name.strip(self.target_estimator.scope)] = sess.run(p)

        save_dict = dict()
        save_dict['q_net'] = q_net_dict
        save_dict['target_net'] = target_net_dict

        np.save(file_name,save_dict)

    # 将训练好的参数导入
    def load_trainable_param_from_file(self,sess,file_name):
        save_dict = np.load(file_name,allow_pickle=True)[()]
        q_net_dict = save_dict['q_net']
        target_net_dict = save_dict['target_net']

        q_net_params = [t for t in tf.trainable_variables() if t.name.startswith(self.q_estimator.scope)]
        q_net_params = sorted(q_net_params, key=lambda v: v.name)

        for p in q_net_params:
            name = p.name
            #print(name)
            assert q_net_dict.__contains__(name.strip(self.q_estimator.scope))
            param_eval = q_net_dict[name.strip(self.q_estimator.scope)]
            opt = p.assign(param_eval)
            sess.run(opt)

        target_net_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target_estimator.scope)]
        target_net_params = sorted(target_net_params, key=lambda v: v.name)

        for p in target_net_params:
            name = p.name
            #print(name)
            assert target_net_dict.__contains__(name.strip(self.target_estimator.scope))
            param_eval = target_net_dict[name.strip(self.target_estimator.scope)]
            opt = p.assign(param_eval)
            sess.run(opt)





class Estimator():
    def __init__(self, scope="estimator", action_num=2, learning_rate=0.001, max_step=50, state_shape=None,
                 mlp_layers=None, lstm_units=10, param_dict=None):
        # param_dict 后续共享参数使用

        # 网络架构[ input->flatten->mlp_layers[0...N-1]->lstm_layers[0...M-1]->action_num]
        self.scope = scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.state_shape = state_shape if isinstance(state_shape, list) else [state_shape]
        self.mlp_layers = mlp_layers
        self.lstm_units = lstm_units
        self.max_time_step = max_step
        self.global_steps = tf.Variable(0, trainable=False)

        with tf.variable_scope(scope):
            self._build_model(param_dict)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name=self.scope + 'drqn_adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)


    def _build_model(self, param_dict):
        # 时间长度 每个样本的时间长度
        self.T_pl = tf.placeholder(shape=[None], dtype=tf.int32, name='T')
        # 构建输入形状
        # 由于是lstm所以应该是[batch_size,time_step,state_shape[...]]
        input_shape = [None, self.max_time_step]  # 第一维为batch,第二维为时间戳
        input_shape.extend(self.state_shape)
        self.X_pl = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")

        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        # 每个样本在最后一个时间点做的action
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        state_flatten_dim = 1
        for s in self.state_shape:
            state_flatten_dim *= s

        # batch norm layer 也包含训练参数
        if param_dict is not None:
            self.batch_norm_layer = param_dict['batch_norm_layer']
        else:
            self.batch_norm_layer = None

        X,self.batch_norm_layer = _batch_normalization(self.X_pl,training=self.is_train,layer = self.batch_norm_layer)

        fc = tf.reshape(X, [-1, self.max_time_step, state_flatten_dim])

        if param_dict is not None:
            self.mlp_fc_dense = param_dict['mlp_fc_dense']
        else:
            self.mlp_fc_dense = []
            for i, dim in enumerate(self.mlp_layers):
                dense = tf.layers.Dense(dim, activation=tf.tanh,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        bias_initializer=tf.zeros_initializer())
                self.mlp_fc_dense.append(dense)

        for dense in self.mlp_fc_dense:
            fc = dense.apply(fc)

        if param_dict is not None:
            self.cell = param_dict['cell']
        else:
            self.cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, activation=tf.tanh)

        batch_size = tf.shape(self.X_pl)[0]
        self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
        self.lstm_output, final_states = tf.nn.dynamic_rnn(cell=self.cell, inputs=fc, initial_state=self.init_state,
                                                           dtype=tf.float32, sequence_length=self.T_pl)

        if param_dict is not None:
            self.predict_dense = param_dict['predict_dense']
        else:
            self.predict_dense = tf.layers.Dense(self.action_num, activation=None,
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 bias_initializer=tf.zeros_initializer())

        # [batch_size,max_step,action_num]
        self.predictions_max_step = self.predict_dense.apply(self.lstm_output)

        # 构建取最后一个时间点的索引
        self.predict_idx = tf.concat([tf.reshape(tf.range(batch_size), [batch_size, 1]),
                                      tf.reshape(tf.subtract(self.T_pl, 1), [batch_size, 1])], axis=1)

        # [batch,action_num]
        self.predictions = tf.gather_nd(self.predictions_max_step, self.predict_idx)
        # self.predictions =

        # 计算loss, 照抄DQN的实现部分
        #                                                 action_num
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        # gather_indices = tf.range(batch_size * self.max_time_step) * self.action_num + tf.reshape(self.actions_pl, [-1])
        # self.action_predictions = tf.reshape(tf.gather(tf.reshape(self.predictions, [-1]), gather_indices),
        # [batch_size, self.max_time_step])
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

    def predict(self, sess, batch_state_seq, batch_length):
        # batch_state_seq:[batch_size,max_step,action_num]
        # batch_length:[batch_size]
        prediction_prob = sess.run(self.predictions, feed_dict={self.X_pl: batch_state_seq, self.T_pl: batch_length,
                                                                self.is_train: False})
        return prediction_prob

    def get_param_dict(self):
        dic = dict()
        dic['scope'] = self.scope
        dic['mlp_fc_dense'] = self.mlp_fc_dense
        dic['cell'] = self.cell
        dic['predict_dense'] = self.predict_dense
        dic['batch_norm_layer'] = self.batch_norm_layer

        return dic

    def update(self, sess, s, a, y, T):
        '''
        :param sess: session
        :param s: batch_stats [batch_size,max_step,state_shape]
        :param a: batch_actions [batch_size]
        :param y: batch_y [batch_size]
        :return: loss
        '''
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.is_train: True, self.T_pl: T}
        gs, _, loss = sess.run(
            [self.global_steps, self.train_op, self.loss],
            feed_dict=feed_dict)

        return loss






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




# return output,batch_norm_layer
def _batch_normalization(inputs,
                            axis=-1,
                            momentum=0.99,
                            epsilon=1e-3,
                            center=True,
                            scale=True,
                            beta_initializer=init_ops.zeros_initializer(),
                            gamma_initializer=init_ops.ones_initializer(),
                            moving_mean_initializer=init_ops.zeros_initializer(),
                            moving_variance_initializer=init_ops.ones_initializer(),
                            beta_regularizer=None,
                            gamma_regularizer=None,
                            beta_constraint=None,
                            gamma_constraint=None,
                            training=False,
                            trainable=True,
                            name=None,
                            reuse=None,
                            renorm=False,
                            renorm_clipping=None,
                            renorm_momentum=0.99,
                            fused=None,
                            virtual_batch_size=None,
                            adjustment=None,
                            layer = None):
    if layer is None:
        layer = tf.layers.BatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            fused=fused,
            trainable=trainable,
            virtual_batch_size=virtual_batch_size,
            adjustment=adjustment,
            name=name,
            _reuse=reuse,
            _scope=name)

    return layer.apply(inputs, training=training),layer


