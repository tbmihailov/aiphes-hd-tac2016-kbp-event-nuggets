import tensorflow as tf

class EventSequenceLabeler_BiLSTM_v2(object):
    def __init__(self,
               n_classes,
               embeddings,
               embeddings_size,
               embeddings_number,
               hidden_size,
               learning_rate,
               learning_rate_trainable=False,
               embeddings_trainable=False):
        # input params

        self.embeddings_size = embeddings_size if embeddings is None else len(embeddings[0])
        self.embeddings_number = embeddings_number if embeddings is None else len(embeddings)
        self.embeddings_trainable = embeddings_trainable

        self.n_classes = n_classes
        self.hidden_size = hidden_size

        # model settings
        self.weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            # 'out': tf.Variable(tf.random_uniform([2*hidden_size, n_classes], minval=-1, maxval=1, dtype=tf.float64), name="out_w", dtype=tf.float64)
            'out': tf.Variable(tf.truncated_normal([2 * self.hidden_size, self.n_classes], stddev=0.1, dtype=tf.float64),
                               trainable=True, name="out_w", dtype=tf.float64)
        }
        self.biases = {
            # 'out': tf.Variable(tf.random_uniform([n_classes], minval=-1, maxval=1, dtype=tf.float64), name="out_b", dtype=tf.float64)
            'out': tf.Variable(tf.truncated_normal([self.n_classes], stddev=0.1, dtype=tf.float64), trainable=True,
                               name="out_b", dtype=tf.float64)
        }

        # input params
        self.input_x = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_y")  # this is not onehot!
        self.input_seq_len = tf.placeholder(dtype=tf.int64, shape=[None], name="input_seq_len")  # length of every sequence in the batch


        with tf.name_scope("embeddings"):
            # embeddings_placeholder = tf.placeholder(tf.float64, shape=[embeddings_number, embedding_size])
            # embeddings_tuned = tf.Variable(embeddings_placeholder)

            # embeddings random, tuned
            # embeddings_tuned =tf.Variable(tf.truncated_normal([embeddings_number, embedding_size], stddev=0.1, dtype=tf.float64), trainable=False, name="embeddings", dtype=tf.float64)

            # embeddings, loaded, tuned
            if not embeddings is None:
                self.embeddings_tuned = tf.Variable(embeddings,
                                               trainable=self.embeddings_trainable,
                                               name="embeddings",
                                               dtype=tf.float64)
            else:
                self.embeddings_tuned = tf.Variable(
                                                tf.truncated_normal([self.embeddings_number, self.embeddings_size],
                                                                    stddev=0.1,
                                                                    dtype=tf.float64),
                                                trainable=self.embeddings_trainable,
                                                name="embeddings", dtype=tf.float64)

            self.embedded_chars = tf.nn.embedding_lookup(self.embeddings_tuned, self.input_x)
            # embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)v# not required for rnn/lstms

        def BiLSTM_dynamic(x_embedd, input_seq_len, shared_fw_bw=False,
                           use_peepholes=False, cell_clip=None,
                           # initializer=None,
                           num_proj=None, proj_clip=None,
                           num_unit_shards=1, num_proj_shards=1,
                           forget_bias=1.0, state_is_tuple=True,
                           activation=tf.tanh):

            with tf.name_scope("BiLSTM"):
                with tf.variable_scope('forward'):
                    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
                                                      use_peepholes=use_peepholes, cell_clip=cell_clip,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=42),
                                                      num_proj=num_proj, proj_clip=proj_clip,
                                                      num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                                                      forget_bias=forget_bias,
                                                      activation=activation)

                with tf.variable_scope('backward'):
                    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True,
                                                      use_peepholes=use_peepholes, cell_clip=cell_clip,
                                                      initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=24),
                                                      num_proj=num_proj, proj_clip=proj_clip,
                                                      num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                                                      forget_bias=forget_bias,
                                                      activation=activation)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_fw if shared_fw_bw else cell_bw,
                    dtype=tf.float64,
                    sequence_length=input_seq_len,
                    inputs=x_embedd)

            return outputs, states

        # Dynamic BiLSTM outputs and states
        outputs, states = BiLSTM_dynamic(x_embedd=self.embedded_chars,
                                         input_seq_len=self.input_seq_len,
                                         cell_clip=None,
                                         shared_fw_bw=True)
        output_fw, output_bw = outputs
        states_fw, states_bw = states

        # we use concatenation over the fw and bw outputs - some approaches use sum?
        # we have to reverse and concat
        output_bw_reversed = tf.reverse(output_bw, dims=(False, False, True))
        outputs_rev = [output_fw, output_bw_reversed]
        output_layer = tf.concat(2, outputs_rev, name="output_layer")

        # Calculate logits and probs
        # Reshape so we can calculate them all at once
        bi_output_concat_flat = tf.reshape(output_layer, [-1, 2 * hidden_size])
        # bi_output_concat_flat_clipped = tf.clip_by_value(bi_output_concat_flat, -1., 1.)  # use clipping if needed. Currently not

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
        self.losses = EventSequenceLabeler_BiLSTM_v2.tf_nan_to_zeros_float64(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat,
                                                                     self.input_y_flat))
        # Replace nans with zeros - yep, there are some Nans from time to time..
        # self.losses = self.losses)

        # Applying the gradients is outside this class!

        print "The model might be okay :D"
        pass

    @staticmethod
    def tf_nan_to_zeros_float64(tensor):
        """
            Mask NaN values with zeros
        :param tensor: Tensor that might have Nan values
        :return: Tensor with replaced Nan values with zeros
        """
        return tf.select(tf.is_nan(tensor), tf.zeros(tf.shape(tensor), dtype=tf.float64), tensor)


