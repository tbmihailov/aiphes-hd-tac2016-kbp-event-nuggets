
"""Event Nugget Detection
    Run train and test on the TAC Event Nugget detection task
"""

from datetime import datetime

import logging  # logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# from sklearn import preprocessing
# from sklearn.grid_search import GridSearchCV
# from sklearn.linear_model import LogisticRegression
from tac_kbp.utils.tf_helpers import tf_helpers
from tac_kbp.utils.Common_Utilities import CommonUtilities

from gensim.models.word2vec import Word2Vec  # used for word2vec

# from sklearn.svm import libsvm
# from sklearn.svm import SVC

# from Word2Vec_AverageVectorsUtilities import AverageVectorsUtilities

from tac_kbp.readers.tac_prep_corpus import *

import time as ti

from tac_kbp.utils.Word2Vec_AverageVectorsUtilities import AverageVectorsUtilities
from tac_kbp.utils.VocabEmbedding_Utilities import VocabEmbeddingUtilities

import pickle
import string
from shutil import copyfile


def count_non_zero_label(test_seq, zero_label=0):
    cnt = 0
    for item in test_seq:
        for lbl in item.y:
            if lbl != zero_label:
                cnt += 1
                # print item.y
                break

    return cnt

def canonicalize_string(str):
    return "".join(c.lower() for c in str if c.isalnum())

def load_eventtypes_canonicalized_from_file(file_name):
    lst = []
    f = codecs.open(file_name, encoding="utf-8")
    for line in f:
        l_canonized = canonicalize_string(line)
        if len(l_canonized) > 0:
            lst.append(l_canonized)

    return lst


def clear_labels_for_eventtypes_notallowed(seq_list, seq_meta_list, allowed_types, zero_y_label, zero_label_str="O"):
    cleared_cnt = 0
    for i, meta_item in enumerate(seq_meta_list):
        for k in range(len(meta_item["labels_type_full"])):
            curr_lbl_typefull = meta_item["labels_type_full"][k]
            if len(curr_lbl_typefull) > 2 and (not canonicalize_string(curr_lbl_typefull[2:]) in allowed_types):
                seq_list[i].y[k] = zero_y_label
                meta_item["labels_event"][k] = zero_label_str
                meta_item["labels_type_full"][k] = zero_label_str
                meta_item["labels_realis"][k] = zero_label_str

                cleared_cnt += 1
    return cleared_cnt

import tensorflow as tf

# from models.EventSequenceLabeler_BiLSTM_v4_posdep_stacked import EventSequenceLabeler_BiLSTM_v4_posdep_stacked
from tac_kbp.models import EventSequenceLabeler_BiLSTM_v3_posdep

from tac_kbp.utils.BatchHelpers import *

class EventLabelerAndClassifier_v4_BiLSTM_v4_posdep_stacked(object):
    def __init__(self,
                 classifier_name,
                 run_name,
                 model_dir,
                 output_dir,
                 event_labeling,
                 event_type_classify,
                 event_realis_classify,
                 event_coref,
                 allowed_event_types):

        self._classifier_name = classifier_name
        self._run_name = run_name
        self._model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self._classifier_settings_dir = "%s/%s" %(model_dir, classifier_name)
        self._vocab_and_embeddings_file = "%s/vocab_and_embeddings.pickle" % (self._classifier_settings_dir)
        self._vocab_and_embeddings_pos_file = "%s/vocab_and_embeddings_pos.pickle" % (self._classifier_settings_dir)
        self._vocab_and_embeddings_deps_file = "%s/vocab_and_embeddings_deps.pickle" % (self._classifier_settings_dir)

        self._output_dir = output_dir

        self._event_labeling = event_labeling
        self._event_type_classify = event_type_classify
        self._event_realis_classify = event_realis_classify
        self._event_coref = event_coref

        self._settings = {}
        self._settings_file = "%s/settings.pickle" % (self._classifier_settings_dir)

        # Checkpoint setup
        self._checkpoint_dir = os.path.abspath(os.path.join(self._classifier_settings_dir, "checkpoints"))
        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "model")
        self._checkpoint_best = os.path.join(self._checkpoint_dir, "model_best")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._checkpoints_backup_dir = os.path.abspath(
            os.path.join(self._classifier_settings_dir, "checkpoints_backup"))
        if not os.path.exists(self._checkpoints_backup_dir):
            os.makedirs(self._checkpoints_backup_dir)

        self.pad_pos = "-PAD-"
        self.unknown_pos = "-UNKNWN-"

        self.pad_deps = "-PAD-"
        self.unknown_deps = "-UNKNWN-"
        self.zero_label_deps = "-NOLBLs-"

        self.unknown_word = "<UNKNWN>"
        self.pad_word = "<PAD>"

        self._allowed_event_types = allowed_event_types
        pass

    def save_settings(self):
        pickle.dump(self._settings, open(self._settings_file, 'wb'))
        logging.info("settings saved to %s" % self._settings_file)

    def load_settings(self):
        self._settings = pickle.load(open(self._settings_file, 'rb'))


    def train(self,
              train_files,
              dev_files,
              embeddings_model,
              embeddings_type,
              embeddings_vec_size,
              # pos_embeddings,
              pos_embeddings_size,
              # pos_embeddings_number,
              # deps_embeddings,
              deps_embeddings_size,
              # deps_embeddings_number,
              eval_dev,
              max_nr_sent,
              embeddings_trainable,
              learning_rate,
              # # learning_rate_trainable = False
              train_epochs_cnt,
              hidden_size,
              batch_size,
              include_pos_layers,
              include_deps_layers,
              include_token_layers,
              lstm_layers,
              learning_rate_fixed=True
              ):

        embeddings_vocab_set = set([])
        if embeddings_type == "w2v":
            embeddings_vocab_set = set(embeddings_model.index2word)

        # Load data

        max_sent_len = 1000

        update_vocab = False
        update_tags = False

        unknown_tag = u'O'
        mapping_file = None
        data_x_fieldname = "tokens"
        data_y_fieldname = "labels_event"
        tag_start_index = 1

        # Retrieve deps vocabulary
        logging.info("Retrieve deps vocabulary..")
        st = ti.time()

        deps_usage_stat = TacPrepJsonCorpus.deps_counts_from_jsonfiles(
            json_files=train_files + dev_files,
            data_fieldname="deps_basic",
            max_nr_sent=max_nr_sent
        )

        deps_vocab = [xx[0] for xx in deps_usage_stat]
        deps_vocab.sort()

        pad_deps = self.pad_deps
        unknown_deps = self.unknown_deps
        zero_label_deps = self.zero_label_deps

        deps_vocab.insert(0, zero_label_deps)
        deps_vocab.insert(0, unknown_deps)
        deps_vocab.insert(0, pad_deps)

        logging.info("deps_vocab:%s" % deps_vocab)

        logging.info("Done in %s s" % (ti.time() - st))

        # Init random POS embeddings
        vocab_and_embeddings_deps = {}

        np.random.seed(111)
        random_embeddings = np.random.uniform(-0.1, 0.1, (len(deps_vocab), deps_embeddings_size))
        deps_vocab_dict_emb = dict([(k, i) for i, k in enumerate(deps_vocab)])

        random_embeddings[deps_vocab_dict_emb[pad_deps]] = np.zeros((deps_embeddings_size))
        vocab_and_embeddings_deps["embeddings"] = random_embeddings
        vocab_and_embeddings_deps["vocabulary"] = deps_vocab_dict_emb

        # save vocab and embeddings
        pickle.dump(vocab_and_embeddings_deps, open(self._vocab_and_embeddings_deps_file, 'wb'))
        logging.info('DEPS: Vocab and deps embeddings saved to: %s' % self._vocab_and_embeddings_deps_file)

        # Retrieve pos vocabulary
        logging.info("Retrieve pos vocabulary..")
        st = ti.time()

        pos_usage_stat = TacPrepJsonCorpus.word_counts_from_jsonfiles(
            json_files=train_files + dev_files,
            data_fieldname="pos",
            max_nr_sent=max_nr_sent
        )

        pos_vocab = [xx[0] for xx in pos_usage_stat]
        pos_vocab.sort()
        pad_pos = self.pad_pos
        unknown_pos = self.unknown_pos
        pos_vocab.insert(0, pad_pos)
        pos_vocab.insert(0, unknown_pos)

        logging.info("pos_vocab:%s" % pos_vocab)

        logging.info("Done in %s s" % (ti.time() - st))

        # Init random POS embeddings
        vocab_and_embeddings_pos = {}

        np.random.seed(111)
        random_embeddings = np.random.uniform(-0.1, 0.1, (len(pos_vocab), pos_embeddings_size))
        pos_vocab_dict_emb = dict([(k, i) for i, k in enumerate(pos_vocab)])

        random_embeddings[pos_vocab_dict_emb[pad_pos]] = np.zeros((pos_embeddings_size))
        vocab_and_embeddings_pos["embeddings"] = random_embeddings
        vocab_and_embeddings_pos["vocabulary"] = pos_vocab_dict_emb

        # save vocab and embeddings
        pickle.dump(vocab_and_embeddings_pos, open(self._vocab_and_embeddings_pos_file, 'wb'))
        logging.info('PoS: Vocab and pos embeddings saved to: %s' % self._vocab_and_embeddings_pos_file)

        # Retrieve words vocabulary
        logging.info("Retrieve words vocabulary..")
        st = ti.time()

        word_usage_stat = TacPrepJsonCorpus.word_counts_from_jsonfiles(
            json_files=train_files,
            data_fieldname=data_x_fieldname,
            max_nr_sent=max_nr_sent
            )

        logging.info("Done in %s s" % (ti.time()-st))

        min_word_freq = 5
        logging.info("min_word_freq:%s" % min_word_freq)
        word_usage_stat = [xx for xx in word_usage_stat if xx[1] >= min_word_freq]

        # clear the vocab
        vocab = [xx[0] for xx in word_usage_stat]

        add_lowercased = True
        if add_lowercased:
            vocab_lowercase = set([xx.lower() for xx in vocab])
            vocab_lowercase = list(vocab_lowercase - set(vocab))
            vocab.extend(vocab_lowercase)

        # Add pad and unknown tokens
        unknown_word = self.unknown_word
        pad_word = self.pad_word

        vocab.insert(0, pad_word)
        vocab.insert(0, unknown_word)

        vocab_dict = LabelDictionary(vocab, start_index=0)

        logging.info("Get average model vector for unknown_vec..")
        st = ti.time()

        vocab_and_embeddings = {}
        if embeddings_type == "w2v":
            unknown_vec = AverageVectorsUtilities.makeFeatureVec(words = list(embeddings_vocab_set),
                                                   model = embeddings_model,
                                                   num_features = embeddings_vec_size,
                                                   index2word_set = embeddings_vocab_set)

            pad_vec = unknown_vec * 0.25
            logging.info("Done in %s s" % (ti.time() - st))

            logging.info("Loading embeddings for vocab..")
            st = ti.time()

            vocab_and_embeddings = VocabEmbeddingUtilities\
                .get_embeddings_for_vocab_from_model(vocabulary=vocab_dict,
                                                   embeddings_type='w2v',
                                                   embeddings_model=embeddings_model,
                                                   embeddings_size=embeddings_vec_size)

            vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][unknown_word], :] = unknown_vec
            vocab_and_embeddings["embeddings"][vocab_and_embeddings["vocabulary"][pad_word], :] = pad_vec

            logging.info("Done in %s s" % (ti.time() - st))

        elif embeddings_type == "rand":
            np.random.seed(123)
            random_embeddings = np.random.uniform(-0.1, 0.1, (len(vocab), embeddings_vec_size))
            vocab_dict_emb = dict([(k, i) for i, k in enumerate(vocab)])

            vocab_and_embeddings["embeddings"] = random_embeddings
            vocab_and_embeddings["vocabulary"] = vocab_dict_emb
        else:
            raise Exception("embeddings_type=%s is not supported!" % embeddings_type)

        # save vocab and embeddings
        pickle.dump(vocab_and_embeddings, open(self._vocab_and_embeddings_file, 'wb'))
        logging.info('Vocab and embeddings saved to: %s' % self._vocab_and_embeddings_file)

        # Load the data for labeling
        corpus_vocab_input = LabelDictionary()
        corpus_vocab_input.set_dict(vocab_and_embeddings["vocabulary"])

        labels_lst = [u'O', u'B-EVENT', u'I-EVENT']
        classes_dict = {1: u'O', 2: u'B-EVENT',  3: u'I-EVENT'}

        self._settings["labels_lst"] = labels_lst
        self._settings["classes_dict"] = classes_dict

        tac_corpus = TacPrepJsonCorpus([], labels_lst,
                                       tag_start_index=1,  # we keep the 0 for padding symbol for Tensorflow dynamic stuff
                                       vocab_start_index=0)

        tac_corpus.set_word_dict(corpus_vocab_input)


        # Load train data
        logging.info("Loading train data from %s..." % train_files)
        st = ti.time()
        train_seq, train_seq_meta = tac_corpus.read_sequence_list_tac_json(train_files,
                                                                           max_sent_len=max_sent_len,
                                                                           max_nr_sent=max_nr_sent,
                                                                           update_vocab=update_vocab,
                                                                           update_tags=update_tags,
                                                                           unknown_word=unknown_word,
                                                                           unknown_tag=unknown_tag,
                                                                           mapping_file=mapping_file,
                                                                           data_x_fieldname=data_x_fieldname,
                                                                           data_y_fieldname=data_y_fieldname)

        logging.info("Clearing labels...to match allowed event tyeps")
        cnt_cleared = clear_labels_for_eventtypes_notallowed(train_seq, train_seq_meta, allowed_types=self._allowed_event_types, zero_y_label=1, zero_label_str="O" )
        logging.info("Cleared labels:%s"%cnt_cleared)

        train_pos = Tac2016_EventNuggets_DataUtilities.get_data_idx_for_field(
                                    data_meta=train_seq_meta,
                                    field_name="pos",
                                    field_vocab_dict=pos_vocab_dict_emb,
                                    unknown_word=unknown_pos)
        logging.info("train_pos[0]:%s" % train_pos[0])

        train_deps_left, train_deps_right = Tac2016_EventNuggets_DataUtilities.get_left_right_data_idx_for_deps(
                                         data_meta=train_seq_meta,
                                         field_name="deps_basic",
                                         field_vocab_dict=deps_vocab_dict_emb,
                                         unknown_lbl=unknown_deps,
                                         zero_deps_lbl=zero_label_deps,
                                         field_sent_tokens="tokens")
        logging.info("train_deps_left[0]:%s" % train_deps_left[0])
        logging.info("train_deps_right[0]:%s" % train_deps_right[0])


        logging.info("Done in %s s" % (ti.time() - st))
        logging.info("All sents:%s" % len(train_seq))
        logging.info("With non zero labels:%s" % count_non_zero_label(train_seq, zero_label=tag_start_index))

        # exit()

        # Load dev data
        logging.info("Loading dev data from %s..." % dev_files)
        st = ti.time()

        dev_seq = None
        dev_seq_meta = None
        if eval_dev and len(dev_files) > 0:

            dev_seq, dev_seq_meta = tac_corpus.read_sequence_list_tac_json(dev_files,
                                                                               max_sent_len=max_sent_len,
                                                                               max_nr_sent=max_nr_sent,
                                                                               update_vocab=update_vocab,
                                                                               update_tags=update_tags,
                                                                               unknown_word=unknown_word,
                                                                               unknown_tag=unknown_tag,
                                                                               mapping_file=mapping_file,
                                                                               data_x_fieldname=data_x_fieldname,
                                                                               data_y_fieldname=data_y_fieldname)

            logging.info("Clearing labels...to match allowed event tyeps")
            cnt_cleared = clear_labels_for_eventtypes_notallowed(dev_seq, dev_seq_meta,
                                                                 allowed_types=self._allowed_event_types,
                                                                 zero_y_label=1, zero_label_str="O")
            logging.info("Cleared labels:%s" % cnt_cleared)

            dev_pos = Tac2016_EventNuggets_DataUtilities.get_data_idx_for_field(
                data_meta=dev_seq_meta,
                field_name="pos",
                field_vocab_dict=pos_vocab_dict_emb,
                unknown_word=unknown_pos)
            logging.info("dev_pos[0]:%s" % dev_pos[0])

            dev_deps_left, dev_deps_right = Tac2016_EventNuggets_DataUtilities.get_left_right_data_idx_for_deps(
                data_meta=dev_seq_meta,
                field_name="deps_basic",
                field_vocab_dict=deps_vocab_dict_emb,
                unknown_lbl=unknown_deps,
                zero_deps_lbl=zero_label_deps,
                field_sent_tokens="tokens")

            logging.info("dev_deps_left[0]:%s" % dev_deps_left[0])
            logging.info("dev_deps_right[0]:%s" % dev_deps_right[0])

            logging.info("Done in %s s" % (ti.time() - st))
            logging.info("All sents:%s" % len(dev_seq))
            logging.info("With non zero labels:%s" % count_non_zero_label(dev_seq, zero_label=tag_start_index))

        else:
            logging.info("No dev evaluation.")
        # TO DO: Train the model!

        logging.info("Train the model..")
        # Train the sequence event labeler
        embeddings = vocab_and_embeddings["embeddings"]
        embeddings_size = vocab_and_embeddings["embeddings"].shape[1]
        embeddings_number = vocab_and_embeddings["embeddings"].shape[0]

        pos_embeddings = vocab_and_embeddings_pos["embeddings"]
        pos_embeddings_size = vocab_and_embeddings_pos["embeddings"].shape[1]
        pos_embeddings_number = vocab_and_embeddings_pos["embeddings"].shape[0]

        deps_embeddings = vocab_and_embeddings_deps["embeddings"]
        deps_embeddings_size = vocab_and_embeddings_deps["embeddings"].shape[1]
        deps_embeddings_number = vocab_and_embeddings_deps["embeddings"].shape[0]

        # embeddings_trainable = False
        # learning_rate = 0.1
        learning_rate_trainable = False
        # train_epochs_cnt = 1

        self._settings["checkpoint_best"] = self._checkpoint_best

        self.save_settings()
        self.train_sequencelabeler_and_save_model(train_seq=train_seq,
                                   train_seq_meta=train_seq_meta,
                                   train_pos=train_pos,
                                   train_deps_left=train_deps_left,
                                   train_deps_right=train_deps_right,
                                   dev_seq=dev_seq,
                                   dev_seq_meta=dev_seq_meta,
                                    dev_pos=dev_pos,
                                    dev_deps_left=dev_deps_left,
                                    dev_deps_right=dev_deps_right,
                                   eval_dev=eval_dev,
                                   n_classes=len(labels_lst),
                                   classes_dict=classes_dict,
                                   embeddings=embeddings,
                                   embeddings_size=embeddings_size,
                                   embeddings_number=embeddings_number,
                                   embeddings_trainable=embeddings_trainable,
                                   pos_embeddings=pos_embeddings,
                                   pos_embeddings_size=pos_embeddings_size,
                                   pos_embeddings_number=pos_embeddings_number,
                                   deps_embeddings=deps_embeddings,
                                   deps_embeddings_size=deps_embeddings_size,
                                   deps_embeddings_number=deps_embeddings_number,
                                   hidden_size=hidden_size,
                                   learning_rate=learning_rate,
                                   learning_rate_trainable=learning_rate_trainable,
                                   train_epochs_cnt=train_epochs_cnt,
                                   batch_size=batch_size,
                                  include_pos_layers=include_pos_layers,
                                  include_deps_layers=include_deps_layers,
                                  include_token_layers=include_token_layers,
                                   learning_rate_fixed=learning_rate_fixed,
                                  lstm_layers=lstm_layers
                                                  )



    def train_sequencelabeler_and_save_model(self,
                              train_seq,
                              train_seq_meta,
                             train_pos,
                             train_deps_left,
                             train_deps_right,
                              dev_seq,
                              dev_seq_meta,
                             dev_pos,
                             dev_deps_left,
                             dev_deps_right,
                              eval_dev,
                              n_classes,
                              classes_dict,
                              embeddings,
                              embeddings_size,
                              embeddings_number,
                              embeddings_trainable,
                               pos_embeddings,
                               pos_embeddings_size,
                               pos_embeddings_number,
                               deps_embeddings,
                               deps_embeddings_size,
                               deps_embeddings_number,
                              hidden_size,
                              learning_rate,
                              learning_rate_trainable,
                              train_epochs_cnt,
                              batch_size,
                              include_pos_layers,
                              include_deps_layers,
                              include_token_layers,
                               learning_rate_fixed,
                              lstm_layers
                              ):


        allow_soft_placement = True
        log_device_placement = True

        # train settings
        pad_value = 0
        # batch_size = 50
        epochs_cnt = train_epochs_cnt  # set to 1 for debug purposes.
        checkpoint_every = 5
        eval_every = 1

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=allow_soft_placement,
                log_device_placement=log_device_placement)

            sess = tf.Session(config=session_conf)
            with sess.as_default():
                seq_model = EventSequenceLabeler_BiLSTM_v3_posdep(
                                       n_classes=n_classes,
                                       embeddings=embeddings,
                                       embeddings_size=embeddings_size,
                                       embeddings_number=embeddings_number,
                                        pos_embeddings=pos_embeddings,
                                        pos_embeddings_size=pos_embeddings_size,
                                        pos_embeddings_number=pos_embeddings_number,
                                        deps_embeddings=deps_embeddings,
                                        deps_embeddings_size=deps_embeddings_size,
                                        deps_embeddings_number=deps_embeddings_number,
                                       hidden_size=hidden_size,
                                       learning_rate=learning_rate,
                                       learning_rate_trainable=learning_rate_trainable,
                                       embeddings_trainable=embeddings_trainable,
                                        include_pos_layers=include_pos_layers,
                                        include_deps_layers=include_deps_layers,
                                        include_token_layers=include_token_layers,
                                       bilstm_layers=lstm_layers
                )

                # We can train also the learning rate
                #learn_rate = tf.Variable(learning_rate, trainable=learning_rate_trainable)
                learn_rate = tf.placeholder(tf.float32, shape=[], name="learn_rate")

                global_step = tf.Variable(0, name="global_step", trainable=False)

                # Compute and apply gradients
                optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)

                gvs = optimizer.compute_gradients(seq_model.losses)

                logging.info("gradients:")
                for grad, var in gvs:
                    logging.info("%s - %s" % (grad, var))
                capped_gvs = [(tf.clip_by_value(tf_helpers.tf_nan_to_zeros_float64(grad), -1., 1.) if grad is not None else grad, var) for grad, var in
                                   gvs]  # cap to prevent NaNs
                # capped_gvs = [(tf.clip_by_value(tf_helpers.tf_nan_to_zeros_float64(grad), -1., 1.), var) for grad, var in
                #                   gvs]  # cap to prevent NaNs

                apply_grads_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                graph_ops = {}
                graph_ops["apply_grads_op"] = apply_grads_op
                # graph_ops["learn_rate"] = learn_rate
                with tf.name_scope("accuracy"):
                    # Calculate the accuracy
                    # Used during training
                    # Mask the losses - padded values are zeros
                    mask = tf.sign(tf.cast(seq_model.input_y_flat, dtype=tf.float64))
                    logging.info("mask:%s" % mask)
                    masked_losses = mask * seq_model.losses

                    # Bring back to [batch, class_num] shape
                    masked_losses = tf.reshape(masked_losses, tf.shape(seq_model.input_y))

                    input_seq_len_float = tf.cast(seq_model.input_seq_len, dtype=tf.float64)

                    # Calculate mean loss - depending on the dynamic number of elements
                    mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / input_seq_len_float
                    mean_loss = tf.reduce_mean(mean_loss_by_example)
                    graph_ops["mean_loss"] = mean_loss
                    # # Evaluate model

                    preds_flat = tf.argmax(seq_model.probs_flat, 1)

                    preds_non_paddings = tf.gather(preds_flat, tf.where(tf.greater(seq_model.input_y_flat, [0])))
                    input_y_non_paddings = tf.gather(seq_model.input_y_flat, tf.where(tf.greater(seq_model.input_y_flat, [0])))

                    preds_y = tf.reshape(preds_flat, tf.shape(seq_model.input_y), name="preds_y")
                    graph_ops["preds_y"] = preds_y

                    correct_pred = tf.equal(preds_non_paddings, input_y_non_paddings)

                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))
                    graph_ops["accuracy"] = accuracy

                def train_step(sess, seq_model, curr_learn_rate, x_batch, y_batch, x_batch_seq_len,
                               x_batch_pos, x_batch_deps, x_batch_deps_len, x_batch_deps_mask):

                    feed_dict = {
                        seq_model.input_x: x_batch,  # batch_data_padded_x,
                        seq_model.input_y: y_batch,  # batch_data_padded_y,
                        seq_model.input_seq_len: x_batch_seq_len,  # batch_data_seqlens
                        seq_model.input_x_pos: x_batch_pos,
                        seq_model.input_x_deps: x_batch_deps,
                        seq_model.input_x_deps_len: x_batch_deps_len,
                        seq_model.input_x_deps_mask: x_batch_deps_mask,
                        learn_rate: curr_learn_rate
                    }

                    _, \
                    step, \
                    res_cost, \
                    res_acc\
                        = sess.run([
                        # graph_ops["learn_rate"],
                        graph_ops["apply_grads_op"],
                        global_step,
                        graph_ops["mean_loss"],
                        graph_ops["accuracy"]
                    ],
                        feed_dict=feed_dict)

                    return res_cost, res_acc # , res_learn_rate

                def dev_step(sess, seq_model, x_batch, y_batch, x_batch_seq_len,
                             x_batch_pos, x_batch_deps, x_batch_deps_len, x_batch_deps_mask):

                    feed_dict = {
                        seq_model.input_x: x_batch,  # batch_data_padded_x,
                        seq_model.input_y: y_batch,  # batch_data_padded_y,
                        seq_model.input_seq_len: x_batch_seq_len,  # batch_data_seqlens
                        seq_model.input_x_pos: x_batch_pos,
                        seq_model.input_x_deps: x_batch_deps,
                        seq_model.input_x_deps_len: x_batch_deps_len,
                        seq_model.input_x_deps_mask: x_batch_deps_mask
                    }

                    step, \
                    res_cost, \
                    res_acc, \
                    res_output_y\
                        = sess.run([
                        global_step,
                        # apply_grads_op, - commented so does not apply gradients
                        graph_ops["mean_loss"],
                        graph_ops["accuracy"],
                        graph_ops["preds_y"]
                    ],
                        feed_dict=feed_dict)

                    # print " Dev cost %2.2f | Dev batch acc %2.2f %% in %s\n" % (res_cost, res_acc, ti.time()-start),

                    return res_output_y, res_cost, res_acc

                # Checkpoint setup
                checkpoint_dir = self._checkpoint_dir
                checkpoint_prefix = self._checkpoint_prefix
                checkpoint_best = self._checkpoint_best

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Training
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=10, keep_checkpoint_every_n_hours=1.0)

                init_vars = tf.initialize_all_variables()
                sess.run(init_vars)

                train_batches_cnt = (len(train_seq) / batch_size) + (0 if len(train_seq) % batch_size == 0 else 1)

                batch_size_dev = 200
                dev_batches_cnt = (len(dev_seq) / batch_size_dev) + (0 if len(dev_seq) % batch_size_dev == 0 else 1)

                logging.info("Train batches:%s" % train_batches_cnt)

                # train with batches
                dev_predictions_history = []
                dev_confusion_matrix_history = []
                dev_gold = []
                best_res_by_lbl = {"B-EVENT": {"f_score": 0.00}}
                current_step = 0
                logging.info("Start training in %s epochs" % epochs_cnt)
                batch_cache_train = []
                batch_cache_dev = []

                learning_rate_current = learning_rate
                for epoch in range(0, epochs_cnt):
                    logging.info("Epoch %d:" % (epoch + 1))
                    start_epoch = ti.time()

                    if not learning_rate_fixed and (epoch+1 > 4):
                        learning_rate_current = learning_rate_current/1.2 # https://arxiv.org/abs/1409.2329 - Recurrent Neural Network Regularization

                    logging.info("Curr learning rate:%s" % learning_rate_current)
                    for i in range(0, train_batches_cnt):
                        # print"Batch %s =================" % (i+1)

                        # Get the batch

                        if len(batch_cache_train)>i:
                            curr_batch_cache = batch_cache_train[i]
                            batch_data_padded_x = curr_batch_cache["batch_data_padded_x"]
                            batch_data_padded_y = curr_batch_cache["batch_data_padded_y"]
                            batch_data_seqlens = curr_batch_cache["batch_data_seqlens"]
                            batch_data_pos = curr_batch_cache["batch_data_pos"]
                            batch_data_deps_left = curr_batch_cache["batch_data_deps_left"]
                            batch_data_deps_left_seqlens = curr_batch_cache["batch_data_deps_left_seqlens"]
                            batch_data_deps_left_masks = curr_batch_cache["batch_data_deps_left_masks"]
                            # logging.info("Loaded batch %s from cache" % i)
                            # print batch_data_padded_x
                        else:
                            all_items_cnt = len(train_seq)
                            batch_from = i * batch_size
                            batch_to = i * batch_size + min(batch_size, all_items_cnt - (i * batch_size))

                            batch_data = train_seq[batch_from: batch_to]
                            batch_data_padded_x, batch_data_padded_y, batch_data_seqlens = prepare_batch_data(data=batch_data)

                            max_batch_sent_len = max(batch_data_seqlens)
                            batch_data_pos = train_pos[batch_from: batch_to]
                            batch_data_pos = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in batch_data_pos]
                            batch_data_deps_left = train_deps_left[batch_from: batch_to]
                            batch_data_deps_left, batch_data_deps_left_seqlens, batch_data_deps_left_masks = pad_and_get_mask(
                                batch_data_deps_left, pad_value=0)

                            batch_data_deps_left_seqlens = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in batch_data_deps_left_seqlens]

                            # print "batch_data_pos:%s" % batch_data_pos
                            # print "batch_data_deps_left:%s" % batch_data_deps_left
                            # print "batch_data_deps_left_seqlens:%s" % batch_data_deps_left_seqlens
                            # print "batch_data_deps_left_masks:%s" % batch_data_deps_left_masks

                            batch_data_pos = np.asarray(batch_data_pos)
                            batch_data_deps_left = np.asarray(batch_data_deps_left)
                            batch_data_deps_left_seqlens = np.asarray(batch_data_deps_left_seqlens)
                            batch_data_deps_left_masks = np.asarray(batch_data_deps_left_masks)

                            # print "batch_data_deps_left_seqlens.shape:" + str(batch_data_deps_left_seqlens.shape)
                            # print "batch_data_pos.shape:" + str(batch_data_pos.shape)
                            # print "batch_data_deps_left.shape:" + str(batch_data_deps_left.shape)
                            # print "batch_data_deps_left_masks.shape:" + str(batch_data_deps_left_masks.shape)

                            curr_batch_cache = {"batch_data_padded_x":batch_data_padded_x,
                                                "batch_data_padded_y":batch_data_padded_y,
                                                "batch_data_seqlens":batch_data_seqlens,
                                                "batch_data_pos":batch_data_pos,
                                                "batch_data_deps_left":batch_data_deps_left,
                                                "batch_data_deps_left_seqlens":batch_data_deps_left_seqlens,
                                                "batch_data_deps_left_masks":batch_data_deps_left_masks}

                            batch_cache_train.append(curr_batch_cache)
                            logging.info("Saved batch %s to cache" % (i))

                        # logging.info("Batch %s: " % (i))
                        # logging.info("batch_data_padded_x.shape:%s " % (str(np.asarray(batch_data_padded_x).shape)))
                        # logging.info("batch_data_padded_y.shape:%s " % (str(np.asarray(batch_data_padded_y).shape)))
                        # logging.info("batch_data_deps_left_seqlens.shape:%s " % (str(np.asarray(batch_data_deps_left_seqlens).shape)))
                        # logging.info("batch_data_pos.shape:%s " % (str(np.asarray(batch_data_pos).shape)))
                        # logging.info("batch_data_deps_left.shape:%s " % (str(np.asarray(batch_data_deps_left).shape)))
                        # logging.info("batch_data_deps_left_masks.shape:%s " % (str(np.asarray(batch_data_deps_left_masks).shape)))

                        # Do the train step
                        start = ti.time()
                        # seq_model.input_x_deps
                        cost, acc = train_step(sess, seq_model,
                                               curr_learn_rate=learning_rate_current,
                                               x_batch=batch_data_padded_x,
                                               y_batch=batch_data_padded_y,
                                               x_batch_seq_len=batch_data_seqlens,
                                               x_batch_pos=batch_data_pos,
                                               x_batch_deps=batch_data_deps_left,
                                               x_batch_deps_len=batch_data_deps_left_seqlens,
                                               x_batch_deps_mask=batch_data_deps_left_masks
                                               )

                        current_step = tf.train.global_step(sess, global_step)

                        # print " Train cost %2.2f | Train batch acc %2.2f %% in %s\n" % (cost, acc, ti.time()-start)
                        # logging.info("learning_rate:%s"%lrate)

                    logging.info(" train epoch time %s " % (ti.time() - start_epoch))

                    if eval_dev and epoch > 0 and (epoch+1) % eval_every == 0:
                        # Dev eval - once per epoch
                        logging.info("Dev set:")
                        input_y_all = []
                        pred_y_all = []

                        for i in range(0, dev_batches_cnt):
                            all_items_cnt = len(train_seq)
                            batch_from = i * batch_size_dev
                            batch_to = i * batch_size_dev + min(batch_size_dev, all_items_cnt - (i * batch_size_dev))

                            if len(batch_cache_dev) > i:
                                curr_batch_cache = batch_cache_dev[i]
                                batch_data_padded_x = curr_batch_cache["batch_data_padded_x"]
                                batch_data_padded_y = curr_batch_cache["batch_data_padded_y"]
                                batch_data_seqlens = curr_batch_cache["batch_data_seqlens"]
                                batch_data_pos = curr_batch_cache["batch_data_pos"]
                                batch_data_deps_left = curr_batch_cache["batch_data_deps_left"]
                                batch_data_deps_left_seqlens = curr_batch_cache["batch_data_deps_left_seqlens"]
                                batch_data_deps_left_masks = curr_batch_cache["batch_data_deps_left_masks"]
                                # logging.info("Loaded batch %s from cache" % i)
                                # print batch_data_padded_x
                            else:
                                batch_data = dev_seq[batch_from: batch_to]
                                batch_data_padded_x, batch_data_padded_y, batch_data_seqlens = prepare_batch_data(data=batch_data)

                                max_batch_sent_len = max(batch_data_seqlens)
                                batch_data_pos = dev_pos[batch_from: batch_to]
                                batch_data_pos = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in batch_data_pos]
                                batch_data_deps_left = dev_deps_left[batch_from: batch_to]
                                batch_data_deps_left, batch_data_deps_left_seqlens, batch_data_deps_left_masks = pad_and_get_mask(
                                    batch_data_deps_left, pad_value=0)

                                batch_data_deps_left_seqlens = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in
                                                                batch_data_deps_left_seqlens]



                                batch_data_pos = np.asarray(batch_data_pos)
                                batch_data_deps_left = np.asarray(batch_data_deps_left)
                                batch_data_deps_left_seqlens = np.asarray(batch_data_deps_left_seqlens)
                                batch_data_deps_left_masks = np.asarray(batch_data_deps_left_masks)

                                curr_batch_cache = {"batch_data_padded_x": batch_data_padded_x,
                                                    "batch_data_padded_y": batch_data_padded_y,
                                                    "batch_data_seqlens": batch_data_seqlens,
                                                    "batch_data_pos": batch_data_pos,
                                                    "batch_data_deps_left": batch_data_deps_left,
                                                    "batch_data_deps_left_seqlens": batch_data_deps_left_seqlens,
                                                    "batch_data_deps_left_masks": batch_data_deps_left_masks}

                                batch_cache_dev.append(curr_batch_cache)
                                logging.info("Saved batch %s to cache" % i)


                            res_pred_y, cost, acc = dev_step(sess, seq_model,
                                                             x_batch=batch_data_padded_x,
                                                             y_batch=batch_data_padded_y,
                                                             x_batch_seq_len=batch_data_seqlens,
                                                             x_batch_pos=batch_data_pos,
                                                             x_batch_deps=batch_data_deps_left,
                                                             x_batch_deps_len=batch_data_deps_left_seqlens,
                                                             x_batch_deps_mask=batch_data_deps_left_masks)

                            #         print batch_data_padded_y[10]
                            #         print res_pred_y[10]

                            for j in range(0, len(batch_data)):
                                input_y_all.extend(batch_data_padded_y[j][:batch_data_seqlens[j]])
                                pred_y_all.extend(res_pred_y[j][:batch_data_seqlens[j]])

                        # eval
                        logging.info("Confusion matrix:")

                        conf_matrix = confusion_matrix(input_y_all, pred_y_all, labels=[1, 2, 3])
                        logging.info("\n"+str(conf_matrix))
                        logging.info("Results by class:")
                        # print_acc_from_conf_matrix(conf_matrix, classes_dict)
                        p_r_f1_acc_by_class_dict = get_prec_rec_fscore_acc_from_conf_matrix(conf_matrix, classes_dict)
                        logging.info("label: (prec, recall, f-score, accuracy)")
                        for k,v in p_r_f1_acc_by_class_dict.iteritems():
                            logging.info("%s:%s" % (k, str(v)))

                        if len(dev_gold) == 0:
                            dev_gold = input_y_all[:]

                        logging.info("Best f-score:%s" % best_res_by_lbl["B-EVENT"]["f_score"])
                        logging.info("Curr f-score:%s" % p_r_f1_acc_by_class_dict["B-EVENT"][2])
                        # acc_by_class_dict = get_acc_from_conf_matrix(conf_matrix, classes_dict)
                        if p_r_f1_acc_by_class_dict["B-EVENT"][2] > best_res_by_lbl["B-EVENT"]["f_score"]:
                            logging.info("Better result - F-Score!")

                            best_res_by_lbl["B-EVENT"]["precision"] = p_r_f1_acc_by_class_dict["B-EVENT"][0]
                            best_res_by_lbl["B-EVENT"]["recall"] = p_r_f1_acc_by_class_dict["B-EVENT"][1]
                            best_res_by_lbl["B-EVENT"]["f_score"] = p_r_f1_acc_by_class_dict["B-EVENT"][2]
                            best_res_by_lbl["B-EVENT"]["accuracy"] = p_r_f1_acc_by_class_dict["B-EVENT"][3]

                            # save
                            path = saver.save(sess, checkpoint_best)

                            logging.info("Saved best model checkpoint to {}\n".format(path))
                            best_res_by_lbl["B-EVENT"]["checkpoint"] = path
                            best_res_by_lbl["B-EVENT"]["confusion_matrix"] = conf_matrix

                            try:
                                logging.info("Trying to backup best_model_file")
                                curr_file = path
                                curr_file_meta = path+".meta"

                                backup_file = path.replace(self._checkpoint_dir, self._checkpoints_backup_dir)
                                backup_file_meta = backup_file+".meta"

                                copyfile(curr_file, backup_file)
                                copyfile(curr_file_meta, backup_file_meta)

                                logging.info("File backuped to %s" % backup_file)
                            except Exception as e:
                                logging.error(e)

                        dev_predictions_history.append(pred_y_all[:])
                        dev_confusion_matrix_history.append(conf_matrix)

                        logging.info(" Dev cost %2.2f | Dev acc %2.2f %% in %s\n" % (cost, acc, ti.time() - start))

                    if (epoch+1) % checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.info("Saved model checkpoint to {}\n".format(path))

                    #     print "Class  : P, R, F-Score:"
                    #     print "B-Event:"
                    #     print precision_recall_fscore_support(input_y_all, pred_y_all, average=None, pos_label=2)
                    #     print "I-Event:"
                    #     print precision_recall_fscore_support(input_y_all, pred_y_all, average=None, pos_label=3)
                    #

                # save last checkpoint
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logging.info("Saved model checkpoint to {}\n".format(path))


                os.remove(checkpoint_best) # debug puprpose
                # Restore best model if not exists
                if not os.path.exists(checkpoint_best):
                    logging.warning("Best model checkpoint does not exists:%s" % checkpoint_best)
                    try:
                        path = checkpoint_best
                        logging.info("Trying to restore best checkpoint file..")
                        curr_file = path
                        curr_file_meta = path + ".meta"

                        backup_file = path.replace(self._checkpoint_dir, self._checkpoints_backup_dir)
                        backup_file_meta = backup_file + ".meta"

                        if not os.path.exists(backup_file):
                            logging.warning("Back up of best model does not exists:%s" % backup_file)

                        copyfile(backup_file, curr_file)
                        copyfile(backup_file_meta, curr_file_meta)

                        logging.info("File backup restored to %s" % curr_file)
                    except Exception as e:
                        logging.error(e)
                # calculate accuracies for last 3 epochs
                #
                # logging.info("Democratic choice across last 3 epochs:")
                # pred_hist = np.array(dev_predictions_history[-3:])
                # pred_hist.astype(int)
                # pred_summary = np.zeros((pred_hist.shape[1], 5))
                # for i in range(pred_hist.shape[0]):
                #     for j in range(pred_hist.shape[1]):
                #         if pred_hist[i][j] in [1, 2, 3]:
                #             pred_summary[j][pred_hist[i][j] - 1] += 1.0
                #
                # pred_summary = pred_summary * [1.001, 1.002, 1.003] # give priority to some classes
                #
                # summ_preds = np.argmax(pred_summary, axis=1)+1
                #
                # conf_matrix = confusion_matrix(dev_gold, summ_preds, labels=[1, 2, 3])
                # print conf_matrix
                # print "Accuracy by class:"
                # print_acc_from_conf_matrix(conf_matrix, classes_dict)


    def eval(self,
             test_files,
             max_nr_sent,
             batch_size,
             output_data_json,
             output_submission_file,
             ):

        # Load data settings
        max_sent_len = 1000

        update_vocab = False
        update_tags = False

        unknown_tag = u'O'
        mapping_file = None
        data_x_fieldname = "tokens"
        data_y_fieldname = "labels_event"
        tag_start_index = 1

        unknown_word = "<UNKNWN>"
        pad_word = "<PAD>"

        # Load vocab and embeddings extracted on training
        vocab_and_embeddings = pickle.load(open(self._vocab_and_embeddings_file, 'rb'))
        logging.info('Vocab and embeddings loaded from: %s' % self._vocab_and_embeddings_file)

        vocab_and_embeddings_pos = pickle.load(open(self._vocab_and_embeddings_pos_file, 'rb'))
        logging.info('POS: Vocab and embeddings loaded from: %s' % self._vocab_and_embeddings_pos_file)

        vocab_and_embeddings_deps = pickle.load(open(self._vocab_and_embeddings_deps_file, 'rb'))
        logging.info('DEPS: Vocab and embeddings loaded from: %s' % self._vocab_and_embeddings_deps_file)

        # Load the data for labeling
        corpus_vocab_input = LabelDictionary()
        corpus_vocab_input.set_dict(vocab_and_embeddings["vocabulary"])

        labels_lst = self._settings["labels_lst"]
        classes_dict = self._settings["classes_dict"]

        tac_corpus = TacPrepJsonCorpus([], labels_lst,
                                       tag_start_index=1,
                                       # we keep the 0 for padding symbol for Tensorflow dynamic stuff
                                       vocab_start_index=0)

        tac_corpus.set_word_dict(corpus_vocab_input)

        # Load test data
        logging.info("Loading test data from %s..." % test_files)
        st = ti.time()
        test_seq, test_seq_meta = tac_corpus.read_sequence_list_tac_json(test_files,
                                                                         max_sent_len=max_sent_len,
                                                                         max_nr_sent=max_nr_sent,
                                                                         update_vocab=update_vocab,
                                                                         update_tags=update_tags,
                                                                         unknown_word=unknown_word,
                                                                         unknown_tag=unknown_tag,
                                                                         mapping_file=mapping_file,
                                                                         data_x_fieldname=data_x_fieldname,
                                                                         data_y_fieldname=data_y_fieldname)

        logging.info("Clearing labels...to match allowed event tyeps")
        cnt_cleared = clear_labels_for_eventtypes_notallowed(test_seq, test_seq_meta,
                                                             allowed_types=self._allowed_event_types,
                                                             zero_y_label=1, zero_label_str="O")
        logging.info("Cleared labels:%s" % cnt_cleared)

        pos_vocab_dict_emb = vocab_and_embeddings_pos["vocabulary"]
        test_pos = Tac2016_EventNuggets_DataUtilities.get_data_idx_for_field(
            data_meta=test_seq_meta,
            field_name="pos",
            field_vocab_dict=pos_vocab_dict_emb,
            unknown_word=self.unknown_pos)
        logging.info("test_pos[0]:%s" % test_pos[0])

        deps_vocab_dict_emb = vocab_and_embeddings_deps["vocabulary"]
        test_deps_left, test_deps_right = Tac2016_EventNuggets_DataUtilities.get_left_right_data_idx_for_deps(
            data_meta=test_seq_meta,
            field_name="deps_basic",
            field_vocab_dict=deps_vocab_dict_emb,
            unknown_lbl=self.unknown_deps,
            zero_deps_lbl=self.zero_label_deps,
            field_sent_tokens="tokens")

        logging.info("test_deps_left[0]:%s" % test_deps_left[0])
        logging.info("test_deps_right[0]:%s" % test_deps_right[0])

        logging.info("Done in %s s" % (ti.time() - st))
        logging.info("All sents:%s" % len(test_seq))
        logging.info("With non zero labels:%s" % count_non_zero_label(test_seq, zero_label=tag_start_index))

        logging.info("Done in %s s" % (ti.time() - st))
        logging.info("Test data all sents:%s" % len(test_seq))
        logging.info("With non zero labels:%s" % count_non_zero_label(test_seq, zero_label=tag_start_index))

        checkpoint_file = self._checkpoint_best
        test_predictions = self.load_model_and_eval_sequencelabels(test_seq=test_seq,
                                                                   checkpoint_file=checkpoint_file,
                                                                   n_classes=len(labels_lst),
                                                                   classes_dict=classes_dict,
                                                                   batch_size=batch_size,
                                                                   test_pos=test_pos,
                                                                   test_deps_left=test_deps_left,
                                                                   test_deps_right=test_deps_right
                                                                 )

        print len(test_predictions)
        # Fill metadata with predictions.
        for i, item in enumerate(test_seq_meta):
            default_label = 1
            pred_labels_safe = [xx if xx in classes_dict else default_label for xx in test_predictions[i]]
            item_str_labels = [classes_dict[xx] for xx in pred_labels_safe]
            item["labels_event"] = item_str_labels

        # Extract events
        event_nuggets_by_docs = Tac2016_EventNuggets_DataUtilities.extract_event_nuggets(test_seq_meta)
        # for doc in event_nuggets_by_docs[:3]:
        #     print doc

        # Json file
        output_json_file = output_data_json
        Tac2016_EventNuggets_DataUtilities.save_data_to_json_file(test_seq_meta, output_json_file)
        logging.info("Processed json saved to %s" % output_json_file)

        # Writing submission file
        system_name="aiphes_hd_t16"
        output_file = "%s/%s_submission_output.tbf" % (self._output_dir, self._run_name)
        Tac2016_EventNuggets_DataUtilities.save_to_output_tbf(output_file, event_nuggets_by_docs, system_name)
        logging.info("Submission file saved to %s" % output_file)


    def load_model_and_eval_sequencelabels(self,
                                           test_seq,
                                           test_pos,
                                           test_deps_left,
                                           test_deps_right,
                                           checkpoint_file,
                                           n_classes,
                                           classes_dict,
                                           batch_size):

        # settings
        pad_value = 0
        # batch_size = 100

        allow_soft_placement = True
        log_device_placement = True

        test_batches_cnt = (len(test_seq) / batch_size) + (0 if len(test_seq) % batch_size ==0 else 1)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=allow_soft_placement,
                log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

            logging.info("Graph operations:")
            for opp in graph.get_operations():
                if opp.name.find("input_") > -1:
                    logging.info(opp.name)
            # get input placeholders
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            input_seq_len = graph.get_operation_by_name("input_seq_len").outputs[0]
            input_x_pos = graph.get_operation_by_name("input_x_pos").outputs[0]
            input_x_deps = graph.get_operation_by_name("input_x_deps").outputs[0]
            input_x_deps_len = graph.get_operation_by_name("input_x_deps_len").outputs[0]
            input_x_deps_mask = graph.get_operation_by_name("input_x_deps_mask").outputs[0]

            # get ouptut layer - we want to use the output representations as features for other stuff
            output_layer = graph.get_operation_by_name("output_layer").outputs[0]

            preds_y = graph.get_operation_by_name("accuracy/preds_y").outputs[0]

            def eval_step(sess, x_batch, y_batch, x_batch_seq_len, x_batch_pos, x_batch_deps, x_batch_deps_len, x_batch_deps_mask):
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    input_seq_len: x_batch_seq_len,
                    input_x_pos: x_batch_pos,
                    input_x_deps: x_batch_deps,
                    input_x_deps_len: x_batch_deps_len,
                    input_x_deps_mask: x_batch_deps_mask
                }

                # res_output_layer, \
                res_output_y \
                    = sess.run(
                    [
                    # output_layer,
                    preds_y],
                    feed_dict=feed_dict)

                return res_output_y  #, res_output_layer

                # Dev eval - once per epoch

            logging.info("Evaluation:")
            input_y_flat = []
            pred_y_flat = []

            test_predicts_all = []

            for i in range(0, test_batches_cnt):
                all_items_cnt = len(test_seq)
                batch_from = i * batch_size
                batch_to = i * batch_size + min(batch_size, all_items_cnt - (i * batch_size))

                batch_data = test_seq[batch_from: batch_to]
                logging.info("Batch %s - %s items" % (i, len(batch_data)))
                batch_data_padded_x, batch_data_padded_y, batch_data_seqlens\
                    = prepare_batch_data(data=batch_data)

                max_batch_sent_len = max(batch_data_seqlens)
                batch_data_pos = test_pos[batch_from: batch_to]
                batch_data_pos = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in batch_data_pos]
                batch_data_deps_left = test_deps_left[batch_from: batch_to]
                batch_data_deps_left, batch_data_deps_left_seqlens, batch_data_deps_left_masks = pad_and_get_mask(
                    batch_data_deps_left, pad_value=0)

                batch_data_deps_left_seqlens = [pad(xx, pad_value=0, to_size=max_batch_sent_len) for xx in
                                                batch_data_deps_left_seqlens]

                batch_data_pos = np.asarray(batch_data_pos)
                batch_data_deps_left = np.asarray(batch_data_deps_left)
                batch_data_deps_left_seqlens = np.asarray(batch_data_deps_left_seqlens)
                batch_data_deps_left_masks = np.asarray(batch_data_deps_left_masks)

                # res_output_y, res_output_layer
                res_output_y = eval_step(sess,
                                         x_batch=batch_data_padded_x,
                                         y_batch=batch_data_padded_y,
                                         x_batch_seq_len=batch_data_seqlens,
                                         x_batch_pos=batch_data_pos,
                                         x_batch_deps=batch_data_deps_left,
                                         x_batch_deps_len=batch_data_deps_left_seqlens,
                                         x_batch_deps_mask=batch_data_deps_left_masks
                                         )
                res_output_y = res_output_y[0]

                #         print batch_data_padded_y[10]
                #         print res_pred_y[10]
                logging.info(res_output_y)
                for j in range(0, len(batch_data)):
                    # flatpred and y for calculating the accuracy
                    input_y_flat.extend(batch_data_padded_y[j][:batch_data_seqlens[j]])
                    pred_y_flat.extend(res_output_y[j][:batch_data_seqlens[j]])

                    # predictions for all samples(sentences)
                    test_predicts_all.append(res_output_y[j][:batch_data_seqlens[j]])

            logging.info("Confusion matrix:")

            conf_matrix = confusion_matrix(input_y_flat, pred_y_flat, labels=[1, 2, 3])
            logging.info("\n"+str(conf_matrix))
            logging.info("F-score by class:")
            acc_by_class_dict = get_prec_rec_fscore_acc_from_conf_matrix(conf_matrix, classes_dict)


            acc_print = "\n"+string.join(["%s : F1=%s, %s" % (k, v[2], v) for k, v in acc_by_class_dict.iteritems()], "\n")

            logging.info(acc_print)

            return test_predicts_all


# Set logging info
logFormatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s]: %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enable file logging
logFileName = '%s/%s-%s.log' % ('logs', 'sup_parser_v1', '{:%Y-%m-%d-%H-%M-%S}'.format(datetime.now()))
# fileHandler = logging.FileHandler(logFileName, 'wb')
# fileHandler.setFormatter(logFormatter)
# logger.addHandler(fileHandler)

# Enable console logging
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# SAMPLE RUN:
# TRAIN:
# run_name="run_v1_tr201415_eval2015"
# output_dir=output/${run_name}
# mkdir -p ${output_dir}

# #model dir where output models are saved after train
# model_dir=models/${run_name}
# rm -rf -- ${model_dir}
# mkdir -p ${model_dir}
#
# scale_features=True

# # resources
# emb_model_type=w2v
# emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
# # emb_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin
#
# word2vec_load_bin=False
# # word2vec_load_bin=True # for google pretrained embeddings

# python tac_kbp/detect_events_v1_bilstm.py -cmd:train -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model}
if __name__ == "__main__":
    # Run parameters
    cmd = 'train'
    cmd = CommonUtilities.get_param_value("cmd", sys.argv, cmd)
    logging.info('cmd:%s' % cmd)

    # run name for output params
    run_name = ""
    run_name = CommonUtilities.get_param_value("run_name", sys.argv, run_name)
    if run_name != "":
        logging.info(('run_name:%s' % run_name))
    else:
        logging.error('Error: missing input file parameter - run_name')
        quit()

    # dir for saving and loading the models
    model_dir = ""
    model_dir = CommonUtilities.get_param_value("model_dir", sys.argv, model_dir)

    # dir for saving output of the parsing
    output_dir = ""
    output_dir = CommonUtilities.get_param_value("output_dir", sys.argv, output_dir)

    logging.info('model_dir:%s' % model_dir)

    model_file_basename = '%s/%s_model_' % (model_dir, run_name)
    scale_file_basename = '%s/%s_scalerange_' % (model_dir, run_name)

    # Input Data
    data_tac2014_train = data_dir+"/clear_data/data_tac2014_train.json"
    data_tac2014_eval = data_dir+"/clear_data/data_tac2014_eval.json"
    data_tac2015_train = data_dir+"/clear_data/data_tac2015_train.json"
    data_tac2015_eval = data_dir+"/clear_data/data_tac2015_eval.json"

    # Settings
    # Perform scaling on the features
    scale_features = False
    scale_features = CommonUtilities.get_param_value_bool("scale_features", sys.argv, scale_features)
    logging.info('scale_features:{0}'.format(scale_features))

    embeddings_trainable = True
    embeddings_trainable = CommonUtilities.get_param_value_bool("emb_train", sys.argv, embeddings_trainable)
    logging.info('embeddings_trainable:{0}'.format(embeddings_trainable))

    embeddings_size = 50
    embeddings_size = CommonUtilities.get_param_value_int("emb_size", sys.argv, embeddings_size)
    logging.info('embeddings_size:{0}'.format(embeddings_size))

    # w2v/doc2vec params
    # word2vec word2vec_model file
    embeddings_model_type = "w2v"  #w2v, dep, rand
    embeddings_model_type = CommonUtilities.get_param_value("emb_model_type", sys.argv, default=embeddings_model_type)
    logging.info('embeddings_model_type:%s' % embeddings_model_type)

    embeddings_model_file = ""  #
    embeddings_model_file = CommonUtilities.get_param_value("emb_model", sys.argv, default=embeddings_model_file)
    logging.info('embeddings_model_file:%s' % embeddings_model_file)

    # load word2vec word2vec_model as binary file
    word2vec_load_bin = False
    word2vec_load_bin = CommonUtilities.get_param_value_bool("word2vec_load_bin", sys.argv, word2vec_load_bin)
    logging.info('word2vec_load_bin:{0}'.format(word2vec_load_bin))

    # allowed event types
    event_types_labels_file = ""  # Used for filtering the labels only for specific event types
    event_types_labels_file = CommonUtilities.get_param_value("event_types_labels_file", sys.argv,
                                                              default=event_types_labels_file)
    logging.info('event_types_labels_file:%s' % event_types_labels_file)

    allowed_event_types = []
    if len(event_types_labels_file) > 0:
        logging.info("Loading event types from %s" % event_types_labels_file)
        allowed_event_types = load_eventtypes_canonicalized_from_file(event_types_labels_file)
        logging.info("Loaded %s event types:\n%s" % (len(allowed_event_types), allowed_event_types))

    # Create the main class
    event_labeling = True
    event_type_classify = False
    event_realis_classify = False
    event_coref = False

    logging.info("Jobs to run:")
    logging.info( "event_labeling:%s" % event_labeling)
    logging.info( "event_type_classify:%s" % event_type_classify)
    logging.info( "event_realis_classify:%s" % event_realis_classify)
    logging.info( "event_coref:%s" % event_coref)

    classifier_name = "event_labeler_BiLSTM_v3_posdeps"
    magic_box = EventLabelerAndClassifier_v4_BiLSTM_v4_posdep_stacked(
                    classifier_name=classifier_name,
                    run_name=run_name,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    event_labeling=event_labeling,
                    event_type_classify=event_type_classify,
                    event_realis_classify=event_realis_classify,
                    event_coref=event_coref,
                    allowed_event_types=allowed_event_types
                )

    max_nr_sent = 1000

    max_nr_sent = CommonUtilities.get_param_value_int("max_sents", sys.argv, max_nr_sent)
    logging.info('max_nr_sent:{0}'.format(max_nr_sent))
    if max_nr_sent == 0:
        max_nr_sent = 1000000 # Current corpus reading method require int..

    batch_size = 50
    batch_size = CommonUtilities.get_param_value_int("batch_size", sys.argv, batch_size)
    logging.info('batch_size:{0}'.format(batch_size))

    lstm_hidden_size = 50
    lstm_hidden_size = CommonUtilities.get_param_value_int("lstm_hidden_size", sys.argv, lstm_hidden_size)
    logging.info('lstm_hidden_size:{0}'.format(lstm_hidden_size))

    lstm_layers = 1
    lstm_layers = CommonUtilities.get_param_value_int("lstm_layers", sys.argv, lstm_layers)
    logging.info('lstm_layers:{0}'.format(lstm_layers))
    if lstm_layers < 1:
        lstm_layers = 1
        logging.warning('lstm_layers should be at least 1. Setting to lstm_layers=1!')

    logging.info('lstm_layers:{0}'.format(lstm_layers))

    deps_embeddings_vec_size = 50
    deps_embeddings_vec_size = CommonUtilities.get_param_value_int("deps_emb_size", sys.argv, deps_embeddings_vec_size)
    logging.info('deps_embeddings_vec_size:{0}'.format(deps_embeddings_vec_size))

    pos_embeddings_vec_size = 50
    pos_embeddings_vec_size = CommonUtilities.get_param_value_int("pos_emb_size", sys.argv, pos_embeddings_vec_size)
    logging.info('pos_embeddings_vec_size:{0}'.format(pos_embeddings_vec_size))

    if cmd == "train":
        logging.info("==========================")
        logging.info("======== TRAINING ========")
        logging.info("==========================")
        embeddings_vec_size = embeddings_size
        if embeddings_model_type == "w2v":
            logging.info("Loading w2v model..")
            if word2vec_load_bin:
                embeddings_model = Word2Vec.load_word2vec_format(embeddings_model_file, binary=True)  # use this for google vectors
            else:
                embeddings_model = Word2Vec.load(embeddings_model_file)
            embeddings_vec_size = embeddings_model.syn0.shape[1]
        elif embeddings_model_type == "rand":
            embeddings_model = None
        else:
            raise Exception("embeddings_model_type=%s is not yet supported!" % embeddings_model_type)

        # train data
        input_data_fileslist_train = [data_tac2014_train, data_tac2015_train, data_tac2014_eval]

        train_data_files = ""  #
        train_data_files = CommonUtilities.get_param_value("train_data_files", sys.argv,
                                                                default=train_data_files)

        if train_data_files == "":
            logging.error("No train_data_files provided. ")
            exit()
        else:
            logging.info('train_data_files:%s' % train_data_files)
            input_data_fileslist_train = train_data_files.split(";")

        # dev data
        eval_dev = True

        input_data_fileslist_dev = [data_tac2015_eval]
        dev_data_files = ""  #
        dev_data_files = CommonUtilities.get_param_value("dev_data_files", sys.argv,
                                                           default=dev_data_files)

        if dev_data_files == "":
            logging.error("No dev_data_files provided. ")
            exit()
        else:
            logging.info('dev_data_files:%s' % dev_data_files)
            input_data_fileslist_dev = dev_data_files.split(";")

        learning_rate = 0.1
        learning_rate = CommonUtilities.get_param_value_float("learning_rate", sys.argv,
                                                                default=learning_rate)
        logging.info('learning_rate:%s' % learning_rate)

        train_epochs_cnt = 6
        train_epochs_cnt = CommonUtilities.get_param_value_int("train_epochs_cnt", sys.argv,
                                                              default=train_epochs_cnt)
        logging.info('train_epochs_cnt:%s' % train_epochs_cnt)

        learning_rate_fixed=True
        learning_rate_fixed = CommonUtilities.get_param_value_bool("learning_rate_fixed", sys.argv,
                                                               default=learning_rate_fixed)
        logging.info('learning_rate_fixed:%s' % learning_rate_fixed)

        # -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}

        learning_rate_divideby = 1.2
        learning_rate_divideby = CommonUtilities.get_param_value_float("learning_rate_divideby", sys.argv,
                                                                   default=learning_rate_divideby)
        logging.info('learning_rate_divideby:%s' % learning_rate_divideby)

        learning_rate_decr_fromepoch = 4
        learning_rate_decr_fromepoch = CommonUtilities.get_param_value_int("learning_rate_decr_fromepoch", sys.argv,
                                                                   default=learning_rate_decr_fromepoch)
        logging.info('learning_rate_decr_fromepoch:%s' % learning_rate_decr_fromepoch)

        # Additional information as input params
        logging.info("Additional info - pos, dep embeddings:")
        # Shortcut Token embeddings on input layer
        incl_tokemb_out = False
        incl_tokemb_out = CommonUtilities.get_param_value_bool("incl_tokemb_out", sys.argv,
                                                                   default=incl_tokemb_out)
        logging.info('incl_tokemb_out:%s' % incl_tokemb_out)

        # POS - Include pos embeddings on input layer
        incl_posemb_in = False
        incl_posemb_in = CommonUtilities.get_param_value_bool("incl_posemb_in", sys.argv,
                                                               default=incl_posemb_in)
        logging.info('incl_posemb_in:%s' % incl_posemb_in)
        # POS - Include pos embeddings on output layer
        incl_posemb_out = False
        incl_posemb_out = CommonUtilities.get_param_value_bool("incl_posemb_out", sys.argv,
                                                              default=incl_posemb_out)
        logging.info('incl_posemb_out:%s' % incl_posemb_out)

        # DEP - Include dep embeddings on input layer
        incl_depemb_in = False
        incl_depemb_in = CommonUtilities.get_param_value_bool("incl_depemb_in", sys.argv,
                                                               default=incl_depemb_in)
        logging.info('incl_depemb_in:%s' % incl_depemb_in)

        # DEP - Include dep embeddings on output layer
        incl_depemb_out = False
        incl_depemb_out = CommonUtilities.get_param_value_bool("incl_depemb_out", sys.argv,
                                                              default=incl_depemb_out)
        logging.info('incl_depemb_out:%s' % incl_depemb_out)

        include_token_layers = {'output_layer': incl_tokemb_out}  # always use on the input
        include_pos_layers = {'input_layer': incl_posemb_in, 'output_layer': incl_posemb_out}
        include_deps_layers = {'input_layer': incl_depemb_in, 'output_layer': incl_depemb_out}

        # # uncommend for hardcore
        # include_token_layers = {'output_layer': False}  # always use on the input
        # include_pos_layers = {'input_layer': True, 'output_layer': True}
        # include_deps_layers = {'input_layer': True, 'output_layer': True}

        magic_box.train(train_files=input_data_fileslist_train,
                        dev_files=input_data_fileslist_dev,
                        embeddings_model=embeddings_model,
                        embeddings_type=embeddings_model_type,
                        embeddings_vec_size=embeddings_vec_size,
                        embeddings_trainable=embeddings_trainable,
                        pos_embeddings_size=pos_embeddings_vec_size,
                        deps_embeddings_size=deps_embeddings_vec_size,
                        eval_dev=eval_dev,
                        max_nr_sent=max_nr_sent,
                        train_epochs_cnt=train_epochs_cnt,
                        learning_rate=learning_rate,
                        hidden_size=lstm_hidden_size,
                        batch_size=batch_size,
                        include_deps_layers=include_deps_layers,
                        include_pos_layers=include_pos_layers,
                        include_token_layers=include_token_layers,
                        learning_rate_fixed=learning_rate_fixed,
                        lstm_layers=lstm_layers)

    elif cmd == "eval":
        logging.info("==========================")
        logging.info("======= EVALUATION =======")
        logging.info("==========================")
        # test data

        input_data_fileslist_test = []

        test_data_files = ""  #
        test_data_files = CommonUtilities.get_param_value("test_data_files", sys.argv,
                                                          default=test_data_files)

        # -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}
        output_data_json = "output_proc_data_%s.json.txt" % run_name  #
        output_data_json = CommonUtilities.get_param_value("output_proc_data_json", sys.argv, default=output_data_json)
        logging.info('output_data_json:%s' % output_data_json)

        output_submission_file = "output_submission_%s.tbf.txt" % run_name  #
        output_submission_file = CommonUtilities.get_param_value("output_submission_file", sys.argv,
                                                                 default=output_submission_file)
        logging.info('output_submission_file:%s' % output_submission_file)


        if test_data_files == "":
            logging.error("No test_data_files provided. ")
            exit()
        else:
            logging.info('test_data_files:%s' % test_data_files)
            input_data_fileslist_test = test_data_files.split(";")

        batch_size_eval=300
        magic_box.load_settings()
        magic_box.eval(test_files=input_data_fileslist_test,
                       max_nr_sent=max_nr_sent,
                       batch_size=batch_size_eval,
                       output_data_json=output_data_json,
                       output_submission_file=output_submission_file,
                       # pos_embeddings_size=pos_embeddings_vec_size,
                       # deps_embeddings_size=deps_embeddings_vec_size,
                       )

    else:
        logging.error("Unknown command:%s" % cmd)

