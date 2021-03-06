import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class RNNDiscriminator(object):
    # input_x
    # input_y
    # dropout_prob
    # ypred_for_auc
    def __init__(self, vocab_size, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token = 0,
                 learning_rate=0.01,nrof_class = 2):
        self.num_emb = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = learning_rate
        self.nrof_class = nrof_class
        self.g_params = []

        with tf.variable_scope('rnn_dicriminator'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim])) #randomly initialize embedding
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

            #####################################################################################################
            # placeholder definition
            self.input_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name='input_x') # sequence of tokens generated by generator
            self.input_y = tf.placeholder(tf.float32, shape=[self.batch_size, self.nrof_class], name='input_y')

            # processed for batch
            with tf.device("/cpu:0"):
                self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.input_x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

            ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
            ta_emb_x = ta_emb_x.unstack(self.processed_x)

            #####################################################################################################
            # create initial state
            self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0 = tf.stack([self.h0, self.h0])


            # When current index i < given_num, use the provided tokens as the input at each time step
            def _g_recurrence_1(i, x_t, h_tm1):
                h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
                x_tp1 = ta_emb_x.read(i)
                return i + 1, x_tp1, h_t


            i, x_t, h_tm1 = control_flow_ops.while_loop(
                cond=lambda i, _1, _2: i < self.sequence_length,
                body=_g_recurrence_1,
                loop_vars=(tf.constant(0, dtype=tf.int32),  #i
                           tf.nn.embedding_lookup(self.g_embeddings, self.start_token), #x_t
                           self.h0))  #h_t
            # last token
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            self.scores = self.g_output_unit(h_t)
            self.ypred_for_auc = tf.nn.softmax(self.scores)

            with tf.name_scope("rnn_loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses)

            with tf.name_scope("rnn_train_op"):
                d_optimizer = tf.train.AdamOptimizer(learning_rate)
                grads_and_vars = d_optimizer.compute_gradients(self.loss, self.g_params)
                self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.nrof_class]))
        self.bo = tf.Variable(self.init_matrix([self.nrof_class]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


class RNNDiscriminator2(object):
    # input_x
    # input_y
    # dropout_prob
    # ypred_for_auc
    def __init__(self, vocab_size, batch_size, emb_dim, hidden_dim,
                 sequence_length, start_token = 0,
                 learning_rate=0.01,nrof_class = 2):
        self.num_emb = vocab_size
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.learning_rate = learning_rate
        self.nrof_class = nrof_class
        self.g_params = []

        with tf.variable_scope('rnn_dicriminator2'):
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim])) #randomly initialize embedding
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_recurrent_unit2 = self.create_recurrent_unit2(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

            #####################################################################################################
            # placeholder definition
            self.input_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length], name='input_x') # sequence of tokens generated by generator
            self.input_y = tf.placeholder(tf.float32, shape=[self.batch_size, self.nrof_class], name='input_y')

            # processed for batch
            with tf.device("/cpu:0"):
                self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.input_x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

            ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
            ta_emb_x = ta_emb_x.unstack(self.processed_x)

            #####################################################################################################
            # create initial state
            self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0 = tf.stack([self.h0, self.h0])


            # When current index i < given_num, use the provided tokens as the input at each time step
            def _g_recurrence_1(i, x_t, h_tmc):
                h_tm1, h_tm2 = tf.unstack(h_tmc)
                h_t1 = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
                h_t_hidden,_ = tf.unstack(h_t1)
                h_t2 = self.g_recurrent_unit2(h_t_hidden, h_tm2)
                h_tout = tf.stack([h_t1, h_t2])
                x_tp1 = ta_emb_x.read(i)
                return i + 1, x_tp1, h_tout


            i, x_t, h_tmc = control_flow_ops.while_loop(
                cond=lambda i, _1, _2: i < self.sequence_length,
                body=_g_recurrence_1,
                loop_vars=(tf.constant(0, dtype=tf.int32),  #i
                           tf.nn.embedding_lookup(self.g_embeddings, self.start_token), #x_t
                           tf.stack([self.h0,self.h0])))  #h_t
            # last token
            h_tm1, h_tm2 = tf.unstack(h_tmc)
            h_t1 = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            h_t_hidden,_ = tf.unstack(h_t1)
            h_t2 = self.g_recurrent_unit2(h_t_hidden, h_tm2)
            self.scores = self.g_output_unit(h_t2)
            self.ypred_for_auc = tf.nn.softmax(self.scores)

            with tf.name_scope("rnn_loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses)

            with tf.name_scope("rnn_train_op"):
                d_optimizer = tf.train.AdamOptimizer(learning_rate)
                grads_and_vars = d_optimizer.compute_gradients(self.loss, self.g_params)
                self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Ui = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uog = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Uc = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_recurrent_unit2(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.Ui2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi2 = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wf2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.Uf2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf2 = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wog2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.Uog2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog2 = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wc2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.Uc2 = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc2 = tf.Variable(self.init_matrix([self.hidden_dim]))
        params.extend([
            self.Wi2, self.Ui2, self.bi2,
            self.Wf2, self.Uf2, self.bf2,
            self.Wog2, self.Uog2, self.bog2,
            self.Wc2, self.Uc2, self.bc2])

        def unit(x, hidden_memory):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi2) +
                tf.matmul(previous_hidden_state, self.Ui2) + self.bi2
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf2) +
                tf.matmul(previous_hidden_state, self.Uf2) + self.bf2
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog2) +
                tf.matmul(previous_hidden_state, self.Uog2) + self.bog2
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc2) +
                tf.matmul(previous_hidden_state, self.Uc2) + self.bc2
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.hidden_dim, self.nrof_class]))
        self.bo = tf.Variable(self.init_matrix([self.nrof_class]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)