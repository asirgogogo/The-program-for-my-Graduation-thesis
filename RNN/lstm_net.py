import tensorflow as tf

# ——————————————————定义神经网络变量——————————————————
def lstmCell(rnn_unit,keep_prob):
    with tf.name_scope("lstmCell"):
        # basicLstm单元
        basicLstm = tf.nn.rnn_cell.BasicRNNCell(rnn_unit)
        # dropout
        drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X,input_size,rnn_unit,lstm_layers,keep_prob):
    with tf.name_scope('LSTM'):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        with tf.name_scope('weights'):
            weights = {
                'in': tf.Variable(tf.random_normal([input_size, rnn_unit]), name="weights_in"),
                'out': tf.Variable(tf.random_normal([rnn_unit, 1]), name="weights_out")
            }
        with tf.name_scope('biases'):
            biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name="biases_in"),
                'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name="biases_in")
            }
        with tf.name_scope("input_rnn"):
            input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
            input_rnn1 = tf.matmul(input, weights['in']) + biases['in']
            input_rnn = tf.reshape(input_rnn1, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
        with tf.name_scope("cell"):
            cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell(rnn_unit,keep_prob) for i in range(lstm_layers)])
        with tf.name_scope("init_state"):
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        with tf.name_scope("output"):
            output = tf.reshape(output_rnn, [-1, rnn_unit])
        # output=output_rnn[-1]
        with tf.name_scope("pred"):
            pred = tf.matmul(output, weights['out']) + biases['out']
    return pred, final_states
