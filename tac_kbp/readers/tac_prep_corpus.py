import codecs
from __builtin__ import staticmethod

import itertools

from tac_kbp.sequences.label_dictionary import LabelDictionary
from tac_kbp.sequences.sequence_list import SequenceList
from tac_kbp.utils.Tac2016_EventNuggets_DataUtilities import Tac2016_EventNuggets_DataUtilities
from os.path import dirname
import os
import numpy as np  # This is also needed for theano=True
from collections import Counter

# from nltk.corpus import brown

# Directory where the data files are located.
data_dir = dirname(__file__) + "/../data/"


def compacify(train_seq, test_seq, dev_seq, default_vocab_dict=[], vocab_start_index=0, default_tag_dict=[], tag_start_index=0, theano=False):
    """
    Create a map for indices that is be compact (do not have unused indices)
    """

    # REDO DICTS
    new_x_dict = LabelDictionary(default_vocab_dict, start_index=vocab_start_index)
    new_y_dict = LabelDictionary(default_tag_dict, start_index=tag_start_index)
    for corpus_seq in [train_seq, test_seq, dev_seq]:
        for seq in corpus_seq:
            for index in seq.x:
                word = corpus_seq.x_dict.get_label_name(index)
                if word not in new_x_dict:
                    new_x_dict.add(word)
            for index in seq.y:
                tag = corpus_seq.y_dict.get_label_name(index)
                if tag not in new_y_dict:
                    new_y_dict.add(tag)

    # REDO INDICES
    # for corpus_seq in [train_seq2, test_seq2, dev_seq2]:
    for corpus_seq in [train_seq, test_seq, dev_seq]:
        for seq in corpus_seq:
            for i in seq.x:
                if corpus_seq.x_dict.get_label_name(i) not in new_x_dict:
                    pass
            for i in seq.y:
                if corpus_seq.y_dict.get_label_name(i) not in new_y_dict:
                    pass
            seq.x = [new_x_dict[corpus_seq.x_dict.get_label_name(i)] for i in seq.x]
            seq.y = [new_y_dict[corpus_seq.y_dict.get_label_name(i)] for i in seq.y]
            # For compatibility with GPUs store as numpy arrays and cats to int
            # 32
            if theano:
                seq.x = np.array(seq.x, dtype='int64')
                seq.y = np.array(seq.y, dtype='int64')
        # Reinstate new dicts
        corpus_seq.x_dict = new_x_dict
        corpus_seq.y_dict = new_y_dict

        # Add reverse indices
        corpus_seq.word_dict = {v: k for k, v in new_x_dict.items()}
        corpus_seq.tag_dict = {v: k for k, v in new_y_dict.items()}

        # SANITY CHECK:
        # These must be the same
    #    tmap  = {v: k for k, v in train_seq.x_dict.items()}
    #    tmap2 = {v: k for k, v in train_seq2.x_dict.items()}
    #    [tmap[i] for i in train_seq[0].x]
    #    [tmap2[i] for i in train_seq2[0].x]

    return train_seq, test_seq, dev_seq


class TacPrepJsonCorpus(object):

    def __init__(self, vocab = [], tag_list = ['O'], tag_start_index=0, vocab_start_index=0):
        # Word dictionary.
        self.word_dict = LabelDictionary(vocab, start_index=vocab_start_index)

        # POS tag dictionary.
        # Initialize noun to be tag zero so that it the default tag.
        self.tag_dict = LabelDictionary(tag_list, start_index=tag_start_index)

        # Initialize sequence list.
        self.sequence_list = SequenceList(self.word_dict, self.tag_dict)

    def set_word_dict(self, labeled_dict):
        self.word_dict = labeled_dict
        self.sequence_list = SequenceList(self.word_dict, self.tag_dict)

    @staticmethod
    def word_counts_from_jsonfiles(json_files, data_fieldname = "tokens", max_nr_sent=0,
                                   tokens_lowercase=False):
        vocab_freq = {}
        nr_sent = 0
        for file_name in json_files:
            data = Tac2016_EventNuggets_DataUtilities.load_data_from_json_file(file_name)

            #for dir_data in data:
            for file_data in data:
                for sent_data in file_data["sentences"]:
                    # print sent_data
                    raw_x = sent_data[data_fieldname]

                    for i in range(0, len(raw_x)):
                        word = raw_x[i] if not tokens_lowercase else raw_x[i].lower()

                        if word not in vocab_freq:
                            vocab_freq[word] = 1
                        else:
                            vocab_freq[word] += 1

                    if len(raw_x) > 1:
                        nr_sent += 1

            if max_nr_sent > 0 and nr_sent >= max_nr_sent:
                break

        word_counts = [(k, v) for k, v in vocab_freq.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)

        return word_counts

    @staticmethod
    def word_counts_from_data_meta(data, data_fieldname="tokens", tokens_lowercase=False):
        vocab_freq = {}
        nr_sent = 0

        for sent_data in data:
            # print sent_data
            raw_x = sent_data[data_fieldname]

            for i in range(0, len(raw_x)):
                word = raw_x[i] if not tokens_lowercase else tokens_lowercase.lower()

                if word not in vocab_freq:
                    vocab_freq[word] = 1
                else:
                    vocab_freq[word] += 1

            if len(raw_x) > 1:
                nr_sent += 1

        word_counts = [(k, v) for k, v in vocab_freq.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)

        return word_counts

    @staticmethod
    def deps_counts_from_jsonfiles(json_files, data_fieldname="deps_cc", max_nr_sent=0):
        vocab_freq = {}

        nr_sent = 0
        for file_name in json_files:
            data = Tac2016_EventNuggets_DataUtilities.load_data_from_json_file(file_name)

            # for dir_data in data:
            for file_data in data:
                for sent_data in file_data["sentences"]:
                    # print sent_data
                    raw_x = sent_data[data_fieldname]

                    for i in range(0, len(raw_x)):
                        word = raw_x[i][0]

                        if word not in vocab_freq:
                            vocab_freq[word] = 1
                        else:
                            vocab_freq[word] += 1

                    if len(raw_x) > 1:
                        nr_sent += 1

            if max_nr_sent > 0 and nr_sent >= max_nr_sent:
                break

        word_counts = [(k, v) for k, v in vocab_freq.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)

        return word_counts

    @staticmethod
    def build_vocabulary(sentences, min_freq=5):

        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] >= min_freq]

        # # Mapping from word to index
        # vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        return vocabulary_inv

    # ----------
    # Reads a json file into a sequence list.
    # ----------
    def read_tac_json_instances(self, input_files, max_sent_len, max_nr_sent,
                                update_vocab=False, update_tags=True,
                                unknown_word = "<UNKNWN>", unknown_tag = "O", mapping_file = "", data_x_fieldname = "tokens",
                                data_y_fieldname = "labels_event",
                                lowercase_data_x=False,
                                filter_sentences_with_labels=None):

        nr_sent = 0
        instances = []
        instances_meta = []
        ex_x = []
        ex_y = []

        nr_types = len(self.word_dict)
        nr_pos = len(self.tag_dict)

        # for dir_data in data:
        for file_name in input_files:
            print file_name
            data = Tac2016_EventNuggets_DataUtilities.load_data_from_json_file(file_name)

            for file_data in data:
                # print len(file_data["sentences"])
                for sent_data in file_data["sentences"]:
                    sent_data["file_id"] = file_data["file_id"]
                    sent_data["file_name"] = file_data["file_name"]

                    raw_x = sent_data[data_x_fieldname]
                    raw_y = sent_data[data_y_fieldname]

                    #check if we should continue
                    filter_curr_sent = False
                    if not filter_sentences_with_labels is None:
                        for lbl_start in filter_sentences_with_labels:
                            for lbl_y in raw_y:
                                if lbl_y.startswith(lbl_start):
                                    filter_curr_sent = True
                                    break
                            if filter_curr_sent:
                                break

                    if filter_curr_sent:
                        continue

                    ex_x = []
                    ex_y = []
                    for i in range(0, len(raw_x)):
                        word = raw_x[i] if not lowercase_data_x else raw_x[i].lower()
                        tag = raw_y[i]

                        if word not in self.word_dict:
                            if not update_vocab:
                                word = unknown_word
                            else:
                                self.word_dict.add(word)
                        if tag not in self.tag_dict:
                            if not update_tags:
                                tag = unknown_tag
                            else:
                                self.tag_dict.add(tag)

                        ex_x.append(word)
                        ex_y.append(tag)

                    if len(ex_x) < max_sent_len and len(ex_x) > 1:
                        # print "accept"
                        nr_sent += 1
                        instances.append([ex_x, ex_y])
                        instances_meta.append(sent_data)
                    # else:
                    #     print ex_x

                    if max_nr_sent >0 and nr_sent >=max_nr_sent:
                        return instances, instances_meta

        return instances, instances_meta

    def read_tac_json_instances_proc_sent_data(self, input_files, max_sent_len, max_nr_sent, update_vocab=False, update_tags=True,
                                unknown_word="<UNKNWN>", unknown_tag="O", mapping_file="", data_x_fieldname="tokens",
                                data_y_fieldname="labels_event",
                                lowercase_data_x=False,
                                filter_sentences_with_labels=None):
        """
        Reads data from json file containing already preprocessed sentence data.
         This method is usually used for evaluation
        :param input_files:
        :param max_sent_len:
        :param max_nr_sent:
        :param update_vocab:
        :param update_tags:
        :param unknown_word:
        :param unknown_tag:
        :param mapping_file:
        :param data_x_fieldname:
        :param data_y_fieldname:
        :return:
        """

        nr_sent = 0
        instances = []
        instances_meta = []
        ex_x = []
        ex_y = []

        nr_types = len(self.word_dict)
        nr_pos = len(self.tag_dict)

        # for dir_data in data:
        for file_name in input_files:
            print file_name
            data = Tac2016_EventNuggets_DataUtilities.load_data_from_json_file(file_name)

            for sent_data in data:
                raw_x = sent_data[data_x_fieldname]
                raw_y = sent_data[data_y_fieldname]

                ex_x = []
                ex_y = []
                for i in range(0, len(raw_x)):
                    word = raw_x[i] if not lowercase_data_x else raw_x[i].lower()
                    tag = raw_y[i]

                    if word not in self.word_dict:
                        if not update_vocab:
                            word = unknown_word
                        else:
                            self.word_dict.add(word)
                    if tag not in self.tag_dict:
                        if not update_tags:
                            tag = unknown_tag
                        else:
                            self.tag_dict.add(tag)

                    ex_x.append(word)
                    ex_y.append(tag)

                if len(ex_x) < max_sent_len and len(ex_x) > 1:
                    # print "accept"
                    nr_sent += 1
                    instances.append([ex_x, ex_y])
                    instances_meta.append(sent_data)
                # else:
                #     print ex_x

                if max_nr_sent > 0 and nr_sent >= max_nr_sent:
                    return instances, instances_meta

        return instances, instances_meta

    @staticmethod
    def read_tac_json_anno_by_doc(input_files, max_nr_sent):
        """
        Exctracts annotation data for doc files

        :param input_files:
        :param max_nr_sent: Max number of sentences
        :return: List of docs with annotations:
            {
                "doc_id":<doc_id>,
                "anno_coref": {
                    "R4": {
                      "key": "R4",
                      "E_arg2": "E32",
                      "E_arg1": "E464",
                      "T_arg1": "T14",
                      "id": "4",
                      "T_arg2": "T12"
                    },
                },
                "anno_nuggets": {
                    "T14": {
                      "text": "Amnesty",
                      "type_full": "Justice_Pardon",
                      "token_spans": [
                        [
                          1391,
                          1398
                        ]
                      ],
                      "tokens": [
                        "Amnesty"
                      ],
                      "subtype": "Pardon",
                      "key": "T14",
                      "realis": "Generic",
                      "type": "Justice",
                      "id": "14"
                    },
                }
            }
        """
        nr_sent = 0

        instances_meta = []

        # for dir_data in data:
        for file_name in input_files:
            print file_name
            data = Tac2016_EventNuggets_DataUtilities.load_data_from_json_file(file_name)

            for file_data in data:
                # print len(file_data["sentences"])
                doc_anno_meta = {}
                doc_anno_meta["file_name"] = file_data["file_name"]
                doc_anno_meta["file_id"] = file_data["file_id"]
                doc_anno_meta["doc_id"] = file_data["file_id"]

                if "anno_nuggets" in file_data:
                    doc_anno_meta["anno_nuggets"] = file_data["anno_nuggets"]

                if "anno_coref" in file_data:
                    doc_anno_meta["anno_coref"] = file_data["anno_coref"]

                instances_meta.append(doc_anno_meta)
                for sent_data in file_data["sentences"]:
                    if max_nr_sent > 0 and nr_sent >= max_nr_sent:
                        return instances_meta

        return instances_meta

    #
    # Read json file and return sequence list
    #
    def read_sequence_list_tac_json(self, train_files,
                                    mapping_file=None,
                                    max_sent_len=100000,
                                    max_nr_sent=100000,
                                    update_vocab=False,
                                    update_tags=True,
                                    unknown_word="<UNKNWN>",
                                    unknown_tag="O",
                                    mapping="",
                                    data_x_fieldname="tokens",
                                    data_y_fieldname="labels_event",
                                    input_is_proc_sent_data=False,
                                    lowercase_data_x=False,
                                    filter_sentences_with_labels=None
                                    ):

        # Build mapping of postags:
        mapping = {}
        if mapping_file is not None:
            for line in open(mapping_file):
                coarse, fine = line.strip().split("\t")
                mapping[coarse.lower()] = fine.lower()

        if input_is_proc_sent_data:
            instance_list, instance_meta_list = self.read_tac_json_instances_proc_sent_data(train_files, max_sent_len, max_nr_sent,
                                                         update_vocab=update_vocab, update_tags=update_tags,
                                                         unknown_word=unknown_word, unknown_tag=unknown_tag,
                                                         mapping_file=mapping_file, data_x_fieldname=data_x_fieldname,
                                                         data_y_fieldname=data_y_fieldname)
        else:
            instance_list, instance_meta_list = self.read_tac_json_instances(train_files, max_sent_len, max_nr_sent,
                                                                             update_vocab=update_vocab,
                                                                             update_tags=update_tags,
                                                                             unknown_word=unknown_word,
                                                                             unknown_tag=unknown_tag,
                                                                             mapping_file=mapping_file,
                                                                             data_x_fieldname=data_x_fieldname,
                                                                             data_y_fieldname=data_y_fieldname)


        seq_list = SequenceList(self.word_dict, self.tag_dict)
        for sent_x, sent_y in instance_list:
            # print sent_x
            seq_list.add_sequence(sent_x, sent_y)
            #seq_list.add_sequence_mask_unknown(sent_x, sent_y, unknown_x_mask=unknown_word, unknown_y_mask=unknown_tag)

        return seq_list, instance_meta_list


    # Dumps a corpus into a file
    def save_corpus(self, dir):
        if not os.path.isdir(dir + "/"):
            os.mkdir(dir + "/")
        word_fn = codecs.open(dir + "word.dic", "w", "utf-8")
        for word_id, word in enumerate(self.int_to_word):
            word_fn.write("%i\t%s\n" % (word_id, word))
        word_fn.close()
        tag_fn = open(dir + "tag.dic", "w")
        for tag_id, tag in enumerate(self.int_to_tag):
            tag_fn.write("%i\t%s\n" % (tag_id, tag))
        tag_fn.close()
        word_count_fn = open(dir + "word.count", "w")
        for word_id, counts in self.word_counts.iteritems():
            word_count_fn.write("%i\t%s\n" % (word_id, counts))
        word_count_fn.close()
        self.sequence_list.save(dir + "sequence_list")

    # Loads a corpus from a file
    def load_corpus(self, dir):
        word_fn = codecs.open(dir + "word.dic", "r", "utf-8")
        for line in word_fn:
            word_nr, word = line.strip().split("\t")
            self.int_to_word.append(word)
            self.word_dict[word] = int(word_nr)
        word_fn.close()
        tag_fn = open(dir + "tag.dic", "r")
        for line in tag_fn:
            tag_nr, tag = line.strip().split("\t")
            if tag not in self.tag_dict:
                self.int_to_tag.append(tag)
                self.tag_dict[tag] = int(tag_nr)
        tag_fn.close()
        word_count_fn = open(dir + "word.count", "r")
        for line in word_count_fn:
            word_nr, word_count = line.strip().split("\t")
            self.word_counts[int(word_nr)] = int(word_count)
        word_count_fn.close()
        self.sequence_list.load(dir + "sequence_list")


