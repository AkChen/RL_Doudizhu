import tensorflow as tf
import numpy as np

a = np.random.random([10,5])
print(a)
a[0] = [1,2,3,4]
print(a)
exit()
X_pl = tf.placeholder(dtype=tf.float32,shape =[None,3,6,5,14])
Len_pl = tf.placeholder(dtype=tf.int32,shape =[None])

shape = tf.shape(X_pl)
input = tf.reshape(X_pl,[-1,3,6*5*14])

lstm_layers = tf.contrib.rnn.BasicLSTMCell(3 ,forget_bias=1.0)

outputs, _ =  tf.nn.dynamic_rnn(cell= lstm_layers, inputs = input, dtype=tf.float32,sequence_length=Len_pl)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

gather_data = np.random.randn(5,10,6)
gather_indices = np.random.randint(0,6,size=[5,10])
print(gather_indices)
data_pl = tf.placeholder(dtype=tf.float32,shape=[None,10,6])
indices_pl = tf.placeholder(dtype=tf.int32,shape=[None,10])
#[
# [[0,0,x],[0,1,x1],...[0,time_step,xt]],
# [[1,0,x],[1,1,x1],...[1,time_step,xt]],
# ]
m_feed_dict = {data_pl:gather_data,indices_pl:gather_indices}
batch_size = tf.shape(data_pl)[0]
idx = tf.range(batch_size*10)*6 + tf.reshape(indices_pl,[-1])
data_flatten = tf.reshape(data_pl,[-1])
print(sess.run(batch_size,feed_dict=m_feed_dict))
print(sess.run(idx,feed_dict=m_feed_dict))
print(sess.run(data_flatten,feed_dict=m_feed_dict))

gather = tf.reshape(tf.gather(tf.reshape(data_pl,[-1]),idx),[batch_size,10])


# batch_size * time_step * 3
#

#转换为N*D



gather_result = sess.run(gather,feed_dict={data_pl:gather_data,indices_pl:gather_indices})

print(gather_result)




#out = sess.run(outputs,feed_dict={X_pl:m_x,Len_pl:m_len})

#print(out)