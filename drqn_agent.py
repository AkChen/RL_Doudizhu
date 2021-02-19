'''DRQN Agent'''

import tensorflow as tf

class Estimator():
    def __init__(self, scope="estimator", action_num=2, learning_rate=0.001,max_time_step = 50,state_shape=None, mlp_layers=None,lstm_units = 10):

        # 网络架构[ input->flatten->mlp_layers[0...N-1]->lstm_layers[0...M-1]->action_num]
        self.scope = scope
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.state_shape = state_shape if isinstance(state_shape, list) else [state_shape]
        self.mlp_layers = map(int, mlp_layers)
        self.lstm_units = lstm_units
        self.max_time_step = max_time_step


        with tf.variable_scope(scope):
            self._build_model()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='dqn_adam')

        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def _build_model(self):
        # 时间长度 每个样本的时间长度
        self.T_pl = tf.placeholder(shape=[None],dtype= tf.int32,name= 'T' )
        # 构建输入形状
        # 由于是lstm所以应该是[batch_size,time_step,state_shape[...]]
        input_shape = [None,self.max_time_step] # 第一维为batch,第二维为时间戳
        input_shape.extend(self.state_shape)
        self.X_pl = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        # 注意[batch_size,time_step]
        self.y_pl = tf.placeholder(shape=[None,self.max_time_step], dtype=tf.float32, name="y")

        self.actions_pl = tf.placeholder(shape=[None,self.max_time_step], dtype=tf.int32, name="actions")

        self.is_train = tf.placeholder(tf.bool, name="is_train")



        state_flatten_dim = 1
        for s in self.state_shape:
            state_flatten_dim *= s

        X = tf.layers.batch_normalization(self.X_pl, training=self.is_train)

        fc = tf.reshape(X,[-1,self.max_time_step,state_flatten_dim])

        for dim in self.mlp_layers:
            fc = tf.contrib.layers.fully_connected(fc, dim, activation_fn=tf.tanh)

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_units)

        self.lstm_output,_= tf.nn.dynamic_rnn(cell=self.cell,inputs = fc,dtype=tf.float32,sequence_length=self.T_pl)

        self.predictions = tf.contrib.layers.fully_connected(self.lstm_output,self.action_num, activation_fn=None)
        # [batch_size,max_step,action_num]

        # 计算loss
        batch_size = tf.shape(self.X_pl)[0]
        gather_indices = tf.range(batch_size*self.max_time_step)*self.action_num + tf.reshape(self.actions_pl,[-1])
        self.action_predictions = tf.reshape(tf.gather(tf.reshape(self.predictions,[-1]),gather_indices),[batch_size,self.max_time_step])

        self.losses = tf.reduce_sum(tf.square(self.y_pl - self.action_predictions),axis=1)
        self.loss = tf.reduce_mean(self.losses)

