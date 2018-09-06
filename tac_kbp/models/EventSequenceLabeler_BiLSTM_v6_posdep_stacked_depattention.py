import tensorflow as tf
from tac_kbp.utils.tf_helpers import tf_helpers
import logging
class EventSequenceLabeler_BiLSTM_v6_posdep_stacked_depattention(object):
    def __init__(self,
                n_classes,
                embeddings,
                embeddings_size,
                embeddings_number,
                pos_embeddings,
                pos_embeddings_size,
                pos_embeddings_number,
                deps_embeddings,
                deps_embeddings_size,
                deps_embeddings_number,
                hidden_size,
                learning_rate,
                learning_rate_trainable=False,
                embeddings_trainable=False,
                include_token_layers={'output_layer': False},
                include_pos_layers={'input_layer': False, 'output_layer': False},
                include_deps_layers={'input_layer': False, 'output_layer': False},
                include_dep_tokens_layers={'input_layer': False, 'output_layer': False},
                bilstm_layers=1):
        # input params
        """
            BiLSTM model with pos and deps inforamtion
        :param n_classes: Number of Y labels
        :param embeddings: Embeddings object - numpy array with shape(embeddings_number, embeddings_number)
        :param embeddings_size: Embeddings vector size
        :param embeddings_number:Embedings vocabulary size
        :param pos_embeddings: DEPS Embeddings object - numpy array with shape(pos_embeddings_number,pos_embeddings_number)
        :param pos_embeddings_size: Embeddings vector size
        :param pos_embeddings_number: Embedings vocabulary size
        :param deps_embeddings: POS Embeddings object - numpy array with shape(deps_embeddings_number,deps_embeddings_number)
        :param deps_embeddings_size: Embeddings vector size
        :param deps_embeddings_number: Embedings vocabulary size
        :param hidden_size: Hidden LSTM size.
        :param learning_rate: Start learning rate
        :param learning_rate_trainable: Trainable learning rate
        :param embeddings_trainable: Train embeddings. Any type - w2v, rand
        :param include_pos_layers: Include POS on layers - List of true false for every layer. 0 layer is embeddigns layer
        :param include_deps_layers: Include DEPS on layers - List of true false for every layer. 0 layer is embeddigns layer
        """
        self.include_pos_layers = include_pos_layers
        self.include_deps_layers = include_deps_layers
        self.include_token_layers = include_token_layers
        self.include_dep_tokens_layers =include_dep_tokens_layers
        logging.info("include_dep_tokens_layers:%s"%str(include_dep_tokens_layers))

        def use_on_layer(include_layers_dict, layer_name):
            return include_pos_layers[layer_name] if layer_name in include_pos_layers else False

        self.embeddings_size = embeddings_size if embeddings is None else len(embeddings[0])
        self.embeddings_number = embeddings_number if embeddings is None else len(embeddings)
        self.embeddings_trainable = embeddings_trainable

        self.n_classes = n_classes
        self.hidden_size = hidden_size


        # input params
        self.input_x = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x")

        self.input_x_pos = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x_pos")
        self.input_x_deps = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="input_x_deps")
        self.input_x_deps_len = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x_deps_len")
        self.input_x_deps_mask = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="input_x_deps_mask")

        self.input_x_dep_tokens = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="input_x_dep_tokens")
        self.input_x_dep_tokens_len = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x_dep_tokens_len")
        self.input_x_dep_tokens_mask = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="input_x_dep_tokens_mask")

        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_y")  # this is not onehot!
        self.input_seq_len = tf.placeholder(dtype=tf.int64, shape=[None], name="input_seq_len")  # length of every sequence in the batch

        # model settings
        self.weights = {}
        self.biases = {}

        with tf.name_scope("embeddings"):
            # embeddings_placeholder = tf.placeholder(tf.float32, shape=[embeddings_number, embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            # embeddings random, tuned
            # embeddings_tuned =tf.Variable(tf.truncated_normal([embeddings_number, embedding_size], stddev=0.1, dtype=tf.float32), trainable=False, name="embeddings", dtype=tf.float32)

            # embeddings, loaded, tuned
            if not embeddings is None:
                self.embeddings_tuned = tf.Variable(embeddings,
                                               trainable=self.embeddings_trainable,
                                               name="embeddings",
                                               dtype=tf.float32)
            else:
                self.embeddings_tuned = tf.Variable(
                                                tf.truncated_normal([self.embeddings_number, self.embeddings_size],
                                                                    stddev=0.1,
                                                                    dtype=tf.float32),
                                                trainable=self.embeddings_trainable,
                                                name="embeddings", dtype=tf.float32)

            self.embedded_chars = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_x)
            # embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)v# not required for rnn/lstms

            # pos_embeddings, loaded, tuned
            if not pos_embeddings is None:
                self.pos_embeddings_tuned = tf.Variable(pos_embeddings,
                                                        trainable=self.embeddings_trainable,
                                                        name="pos_embeddings",
                                                        dtype=tf.float32)
            else:
                self.pos_embeddings_tuned = tf.Variable(
                    tf.truncated_normal([self.pos_embeddings_number, self.pos_embeddings_size],
                                        stddev=0.1,
                                        dtype=tf.float32),
                    trainable=self.embeddings_trainable,
                    name="pos_embeddings", dtype=tf.float32)

            self.embedded_pos = tf.nn.embedding_lookup(self.pos_embeddings_tuned, self.input_x_pos)

            # deps_embeddings, loaded, tuned
            if not deps_embeddings is None:
                self.deps_embeddings_tuned = tf.Variable(deps_embeddings,
                                                         trainable=self.embeddings_trainable,
                                                         name="deps_embeddings",
                                                         dtype=tf.float32)
            else:
                self.deps_embeddings_tuned = tf.Variable(
                    tf.truncated_normal([self.deps_embeddings_number, self.deps_embeddings_size],
                                        stddev=0.1,
                                        dtype=tf.float32),
                    trainable=self.embeddings_trainable,
                    name="deps_embeddings", dtype=tf.float32)

            self.embedded_deps = tf.nn.embedding_lookup(self.deps_embeddings_tuned, self.input_x_deps, name="embedded_deps_lookup")
            self.embedded_deps_masked = self.embedded_deps * tf.cast(tf.expand_dims(self.input_x_deps_mask, -1), dtype=tf.float32)
            input_x_deps_len_red = tf.cast(tf.expand_dims(self.input_x_deps_len, -1), dtype=tf.float32)
            self.embedded_deps_reduce_mean = tf.reduce_sum(self.embedded_deps_masked,
                                                           reduction_indices=-2) #  / input_x_deps_len_red - accumulate instead of average
            # self.embedded_deps_reduce_mean_masked = self.embedded_deps_reduce_mean *

            # Dep Tokens
            self.embedded_dep_tokens = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_x_dep_tokens,
                                                        name="embedded_dep_tokens_lookup")
            self.embedded_dep_tokens_masked = self.embedded_dep_tokens * tf.cast(tf.expand_dims(self.input_x_dep_tokens_mask, -1),
                                                                     dtype=tf.float32)

            self.embedded_dep_tokens_reduce_mean = tf.reduce_sum(self.embedded_dep_tokens_masked,
                                                           reduction_indices=-2)
        self.embedded_input_layer = self.embedded_chars
        if use_on_layer(include_pos_layers, 'input_layer'):
            self.embedded_input_layer = tf.concat(2, [self.embedded_input_layer, self.embedded_pos])
            logging.info('input_layer: use POS label embeddings ')
        else:
            logging.info('input_layer: NOT use POS label embeddings ')

        if use_on_layer(include_deps_layers, 'input_layer'):
            self.embedded_input_layer = tf.concat(2, [self.embedded_input_layer, self.embedded_deps_reduce_mean])
            logging.info('input_layer: use DEPS label embeddings sum')
        else:
            logging.info('input_layer: NOT use DEPS label embeddings sum')

        if use_on_layer(include_dep_tokens_layers, 'input_layer'):
            self.embedded_input_layer = tf.concat(2, [self.embedded_input_layer, self.embedded_dep_tokens_reduce_mean])
            logging.info('input_layer: use dependency attention label embeddings sum')
            print "input_layer: use dependency attention label embeddings sum"
        else:
            logging.info('input_layer: NOT use dependency attention label embeddings sum')

        def BiLSTM_dynamic(x_embedd, input_seq_len, shared_fw_bw=False,
                           use_peepholes=False, cell_clip=None,
                           # initializer=None,
                           num_proj=None, proj_clip=None,
                           num_unit_shards=1, num_proj_shards=1,
                           forget_bias=1.0, state_is_tuple=True,
                           activation=tf.tanh,
                           scope_id=0,
                           lstm_layers=1):

            with tf.variable_scope("BiLSTM-%s" % i):
                with tf.variable_scope('forward-%s' % scope_id, reuse=True):
                    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
                                                      use_peepholes=use_peepholes, cell_clip=cell_clip,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=42),
                                                      num_proj=num_proj, proj_clip=proj_clip,
                                                      num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                                                      forget_bias=forget_bias,
                                                      activation=activation)

                with tf.variable_scope('backward-%s' % scope_id, reuse=True):
                    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
                                                      use_peepholes=use_peepholes, cell_clip=cell_clip,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=24),
                                                      num_proj=num_proj, proj_clip=proj_clip,
                                                      num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                                                      forget_bias=forget_bias,
                                                      activation=activation)

                multicell_fw = cell_fw
                multicell_bw = cell_bw

                if lstm_layers>1:
                    multicell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * lstm_layers, state_is_tuple=True)
                    multicell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * lstm_layers, state_is_tuple=True)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=multicell_fw,
                    cell_bw=multicell_fw if shared_fw_bw else multicell_bw,
                    dtype=tf.float32,
                    sequence_length=input_seq_len,
                    inputs=x_embedd)

            return outputs, states

        # Dynamic BiLSTM outputs and states

        output_layer = None
        output_layer_size = 0
        bilstm_layers_cnt = 1  # DUMMY SETTING - TO DO : Refactor

        for i in range(0, bilstm_layers_cnt):
            outputs, states = BiLSTM_dynamic(# x_embedd=self.embedded_chars,
                                             x_embedd=self.embedded_input_layer if i == 0 else output_layer,
                                             input_seq_len=self.input_seq_len,
                                             cell_clip=None,
                                             shared_fw_bw=True,
                                             scope_id=i,
                                             lstm_layers=bilstm_layers)
            output_fw, output_bw = outputs
            # states_fw, states_bw = states

            # we use concatenation over the fw and bw outputs - some approaches use sum?
            # we have to reverse and concat
            output_bw_reversed = tf.reverse(output_bw, dims=(False, False, True))
            outputs_rev = [output_fw, output_bw_reversed]
            output_layer = tf.concat(2, outputs_rev, name="output_layer")
            logging.info('BiLSTM-%s - built' % i)
            output_layer_size = 2 * hidden_size

        # shortcut - use POS and deps embeddings on the output layer

        if use_on_layer(include_token_layers, 'output_layer'):
            output_layer = tf.concat(2, [output_layer, self.embedded_chars])
            logging.info('output_layer: use Word Embeddings')
            output_layer_size += embeddings_size

        if use_on_layer(include_pos_layers, 'output_layer'):
            output_layer = tf.concat(2, [output_layer, self.embedded_pos])
            logging.info('output_layer: use POS label embeddings ')
            output_layer_size += pos_embeddings_size

        if use_on_layer(include_deps_layers, 'output_layer'):
            output_layer = tf.concat(2, [output_layer, self.embedded_deps_reduce_mean])
            logging.info('output_layer: use DEPS label embeddings sum')
            output_layer_size += deps_embeddings_size

        if use_on_layer(include_dep_tokens_layers, 'output_layer'):
            output_layer = tf.concat(2, [output_layer, self.embedded_dep_tokens_reduce_mean])
            logging.info('output_layer: use deps attention label embeddings sum')
            output_layer_size += embeddings_size

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        bi_output_concat_flat = tf.reshape(output_layer, [-1, output_layer_size])
        # bi_output_concat_flat_clipped = tf.clip_by_value(bi_output_concat_flat, -1., 1.)  # use clipping if needed. Currently not


        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        self.weights['out'] = tf.Variable(tf.random_uniform([output_layer_size, n_classes], minval=-0.1, maxval=0.1, dtype=tf.float32), name="out_w", dtype=tf.float32)
        self.biases['out'] = tf.Variable(tf.random_uniform([n_classes], minval=-0.1, maxval=0.1, dtype=tf.float32), name="out_b", dtype=tf.float32)

        self.logits_flat = tf.batch_matmul(bi_output_concat_flat, self.weights["out"]) + self.biases["out"]
        # logits_flat = tf.clip_by_value(logits_flat, -1., 1.)  # use clipping if needed. Currently not
        self.probs_flat = tf.nn.softmax(self.logits_flat)

        print "outputs[-1].shape:%s" % outputs[-1]  #
        print "weights[\"out\"]].shape:%s" % self.weights["out"]  # .shape
        print "biases[\"out\"].shape:%s" % self.biases["out"]  # .shape

        # print pred

        # make y flat so it match pred shape
        self.input_y_flat = tf.reshape(self.input_y, [-1])
        print "input_y_flat:%s" % self.input_y_flat

        # # Define loss and optimizer

        # sparse_softmax_cross_entropy_with_logits - calculates on non-onehot y!
        # this should also not calculate the cross entropy for 0 labels (the padded labels)
        self.losses = tf_helpers.tf_nan_to_zeros_float32(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat,
                                                                     self.input_y_flat))
        # Replace nans with zeros - yep, there are some Nans from time to time..
        # self.losses = self.losses)

        # Applying the gradients is outside this class!

        print "The model might be okay :D"
        pass




