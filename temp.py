# Find the layers
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os
from config import HIDDEN_DIM

def get_all_variables_from_top_scope(scope):
    # scope is a top scope here, otherwise change startswith part
    return [v for v in tf.global_variables() if v.name.startswith(scope)]


def create_recurrent_unit( graph):
    # Weights and Bias for input and hidden tensor
    Wi = graph.get_tensor_by_name('generator/Variable_1:0')
    Ui = graph.get_tensor_by_name('generator/Variable_2:0')
    bi = graph.get_tensor_by_name('generator/Variable_3:0')

    Wf = graph.get_tensor_by_name('generator/Variable_4:0')
    Uf = graph.get_tensor_by_name('generator/Variable_5:0')
    bf = graph.get_tensor_by_name('generator/Variable_6:0')

    Wog = graph.get_tensor_by_name('generator/Variable_7:0')
    Uog = graph.get_tensor_by_name('generator/Variable_8:0')
    bog = graph.get_tensor_by_name('generator/Variable_9:0')

    Wc = graph.get_tensor_by_name('generator/Variable_10:0')
    Uc = graph.get_tensor_by_name('generator/Variable_11:0')
    bc = graph.get_tensor_by_name('generator/Variable_12:0')

    def unit(x, hidden_memory_tm1):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

        # Input Gate
        i = tf.sigmoid(
            tf.matmul(x, Wi) +
            tf.matmul(previous_hidden_state, Ui) + bi
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, Wf) +
            tf.matmul(previous_hidden_state, Uf) + bf
        )

        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, Wog) +
            tf.matmul(previous_hidden_state, Uog) + bog
        )

        # New Memory Cell
        c_ = tf.nn.tanh(
            tf.matmul(x, Wc) +
            tf.matmul(previous_hidden_state, Uc) + bc
        )

        # Final Memory cell
        c = f * c_prev + i * c_

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    return unit


def create_output_unit(graph):
    Wo = graph.get_tensor_by_name('generator/Variable_13:0')
    bo = graph.get_tensor_by_name('generator/Variable_14:0')


    def unit(hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        # hidden_state : batch x hidden_dim
        logits = tf.matmul(hidden_state, Wo) + bo
        # output = tf.nn.softmax(logits)
        return logits
    return unit


def main():
    batch_size =1
    hidden_dim =HIDDEN_DIM
    eof_symbol =96
    fname = 'model12'
    model_path = './Model/'+fname+'/'
    meta_path = model_path +fname+'.meta'
    ckpt_path = model_path +fname+'-5'
    dictionary_file = './Data/dict/TrainableMidi2-Table.txt'
    tgt_prefix = './Data/result/'+fname+'_'
    f_idx =1
    tgt_fpath = tgt_prefix+str(f_idx)+'.txt'
    while os.path.exists(tgt_fpath):
        f_idx += 1
        tgt_fpath = tgt_prefix+str(f_idx)+'.txt'

    with open(dictionary_file, 'r') as f:
        buf = f.read()

    forward_dict = eval(buf)
    backward_dict = {}
    for x,y in forward_dict.items():
        backward_dict[y] = x

    g = tf.Graph()
    with g.as_default():
        saver = tf.train.import_meta_graph(meta_path)

        # l_var2 = get_all_variables_from_top_scope('generator')

        g_embeddings = g.get_tensor_by_name('generator/Variable:0')
        g_recurrent_unit = create_recurrent_unit(g)  # maps h_tm1 to h_t for generator
        g_output_unit = create_output_unit(g)  # maps h_t to o_t (output token logits)

        h0 = tf.zeros([batch_size, hidden_dim])
        h0 = tf.stack([h0, h0])

        # gen_o = tensor_array_ops.TensorArray(dtype=tf.float32,dynamic_size=True, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=32, dynamic_size=True, infer_shape=True)

        def _g_recurrence(i, q, x_t, h_tm1,  gen_x):
            h_t = g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            h_t = tf.Print(h_t,[q])
            o_t = g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i+1, next_token[0], x_tp1, h_t,  gen_x

        _, _, _, _, gen_x = control_flow_ops.while_loop(
            cond=lambda i, q,_1, _2, _3: q<tf.constant(96),
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(g_embeddings, tf.constant([0] * batch_size, dtype=tf.int32)), h0,  gen_x))

        gen_x = gen_x.stack()
        gen_x = tf.transpose(gen_x, perm=[1, 0])  # batch_size x seq_length
        with tf.Session() as sess:
            saver.restore(sess,ckpt_path)
            outputs = sess.run(gen_x)

        outputs = outputs[0,:]
        with open(tgt_fpath, 'w') as f:
            for tok in outputs:
                if tok == eof_symbol:
                    break
                f.write(backward_dict[int(tok)])


if __name__ == '__main__':
    main()
