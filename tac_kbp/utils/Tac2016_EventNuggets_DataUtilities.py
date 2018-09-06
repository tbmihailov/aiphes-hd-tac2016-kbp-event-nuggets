import xml.etree.ElementTree as ET
import xml.etree.cElementTree as cET
import sys
from __builtin__ import staticmethod

import datetime
from tac_kbp.utils.Common_Utilities import CommonUtilities
import time

import json
import os
import codecs
from stanford_corenlp_pywrapper import CoreNLP
import re
import string
import logging

def clear_matches_with_pos_preserve(line_str, re_match_op, repl_str):

    m_lst = re_match_op.finditer(line_str)
    line_out = line_str[:]
    for m in m_lst:
        match_span = m.span()
        repl_str_padded = repl_str +" "*(match_span[1] - match_span[0] - len(repl_str))

        # print "%s - %s" % (m.span())
        line_out = line_out[:match_span[0]]+repl_str_padded+line_out[match_span[1]:]
        # print line_out
    return line_out

class Tac2016_EventNuggets_DataUtilities(object):

    @staticmethod
    def read_idents_from_file(output_file):
        res_data = []
        with open(output_file, "r") as f:
            for line in f:
                res_data.append(line.replace("\n", "").split("\t"))

        return res_data

    @staticmethod
    def load_data_from_json_file(json_file):
        data_file = codecs.open(json_file, mode='r', encoding="utf-8")
        data = json.load(data_file)
        data_file.close()

        return data

    @staticmethod
    def save_data_to_json_file(data, output_json_file):
        data_file = codecs.open(output_json_file, mode='wb', encoding="utf-8")
        json.dump(data, data_file)
        data_file.close()

    @staticmethod
    def tacdata_load_annotations_events_from_brat_format(file_name_annoation, read_coref=False):
        """
            Loads annotation data from brat format
            Args:
                file_name_annoation:
                    Brat annotation file. Used in TAC2014. Data from TAC 2015 should be converted with
            Returns:
                Parsed annotations as dictionary
        """
        print "Coref version"
        print file_name_annoation
        # T5	Conflict_Attack 1246 1253	stoning
        # E5	Conflict_Attack:T5
        # A5	Realis E5 Generic
        # T7	Conflict_Attack 1458 1462;1468 1472;1483 1489	turn into crater
        # E7	Conflict_Attack:T7
        # A7	Realis E7 Other
        # R1	Coreference Arg1:E207 Arg2:E194
        # R2	Coreference Arg1:E194 Arg2:E220
        # R3	Coreference Arg1:E233 Arg2:E246

        t_items = {}  # tokens
        e_items = {}  # type
        a_items = {}  # realis
        r_items = {}  # coreference

        f_ann = codecs.open(file_name_annoation, mode='rb', encoding = 'utf-8')
        for line in f_ann:
            try:
                str = line.strip()
                cols = str.split('\t')
                key = cols[0]
                id = cols[0][1:]
                key_type = cols[0][:1]
                if key_type == "T":
                    row_data = {}
                    row_data["id"] = id
                    row_data["key"] = key
                    row_data["text"] = cols[2]

                    col1_items = cols[1].split(" ", 1)
                    # type
                    row_data["type_full"] = col1_items[0]
                    if "_" in row_data["type_full"]:
                        row_data["type"] = col1_items[0].split("_")[0]
                        row_data["subtype"] = col1_items[0].split("_")[1]
                    else:
                        row_data["type"] = row_data["type_full"]
                        row_data["subtype"] = ""
                    # spans
                    spans = [(int(y[0]), int(y[1])) for y in [x.split(' ') for x in col1_items[1].split(';')]]

                    text_tokens = row_data["text"].split(" ")
                    if len(text_tokens)>1 and len(spans)==1:
                        # split on tokens
                        span_begin = spans[0][0]
                        span_end = spans[0][1]

                        spans_new = []
                        curr_loc = span_begin
                        for ttoken in text_tokens:
                            spans_new.append((curr_loc, curr_loc+len(ttoken)-1))
                            curr_loc += len(ttoken)+1  # this includes +1 for skipping space

                        # print "spans handled for %s: %s -> %s" % (row_data["text"],spans, spans_new)
                        spans = spans_new
                    elif len(text_tokens) > len(spans):
                        span_begin = spans[0][1]
                        spans_new = []

                        txt_curr = row_data["text"]
                        for sp in spans:
                            sp_txt = txt_curr[sp[0]-span_begin:sp[1]-span_begin]
                            if " " in sp_txt:
                                sp_txt_subtokens = sp_txt.split(" ")
                                curr_loc = sp[0]
                                for ttoken in sp_txt_subtokens:
                                    spans_new.append((curr_loc, curr_loc + len(ttoken) - 1))
                                    curr_loc += len(ttoken) + 1  # this includes +1 for skipping space
                            else:
                                spans_new.append((sp[0], sp[1]))
                        # print "2 - spans handled for %s: %s -> %s" % (row_data["text"], spans, spans_new)

                    elif len(text_tokens) < len(spans):
                        raise Exception("Token-span mismatch at %s in file %s:\n%s"%(cols[0], file_name_annoation, str))

                    row_data["token_spans"] = spans
                    row_data["tokens"] = row_data["text"].split(" ")

                    t_items[key] = row_data
                elif key_type == "E":
                    row_data_e = {}
                    col1_items = cols[1].split(":")
                    row_data_e["T"] = col1_items[1]
                    row_data_e["type_full"] = col1_items[0]
                    e_items[key] = row_data_e
                elif key_type == "A":
                    row_data_a = {}
                    col1_items = cols[1].split(" ")
                    row_data_a["E"] = col1_items[1]
                    if col1_items[0] == "Realis":
                        a_items[key] = row_data_a
                        row_data_a["realis"] = col1_items[2]
                    else:
                        row_data_a["realis"] = "Questionable"
                        print "Non realis:Wrong format at %s in file %s:\n%s" % (cols[0], file_name_annoation, str)

                elif read_coref and key_type == "R":  # and cols[1] == "Coreference":
                    # R1	Coreference Arg1:E207 Arg2:E194
                    # R2	Coreference Arg1:E194 Arg2:E220
                    # R3	Coreference Arg1:E233 Arg2:E246

                    row_data_r = {}
                    row_data_r["id"] = id
                    row_data_r["key"] = key

                    if len(cols)<=1:
                        print "Invalid coreference row: %s" % str
                    else:
                        data_coref_args = cols[1].split(" ")
                        arg1_event_key = data_coref_args[1].split(":")[1]
                        arg2_event_key = data_coref_args[2].split(":")[1]

                        row_data_r["E_arg1"] = arg1_event_key
                        row_data_r["E_arg2"] = arg2_event_key

                        r_items[key] = row_data_r
                        print row_data_r
            except Exception as ex:
                print ex
                pass

        # update event info with realis
        for k,v in a_items.iteritems():
            if v["E"] in e_items:
                t_key = e_items[v["E"]]["T"]
                if t_key in t_items:
                    t_items[t_key]["realis"] = v["realis"]

        if read_coref:
            for k, v in r_items.iteritems():
                try:
                    t_key_arg1 = e_items[v["E_arg1"]]["T"]
                    t_key_arg2 = e_items[v["E_arg2"]]["T"]

                    v["T_arg1"] = t_key_arg1
                    v["T_arg2"] = t_key_arg2
                except Exception as ex:
                    print ex
                    pass


        if read_coref:
            return t_items, r_items
        else:
            return t_items

    @staticmethod
    def get_tokens_with_annotations(data_annotation):
        tokens_with_labels = []
        tokens_with_labels_spantree = {}
        for k, v in data_annotation.iteritems():
            token_spans = v["token_spans"]
            for i in range(len(token_spans)):
                token_data = {}
                token_data["begin"] = token_spans[i][0]
                token_data["end"] = token_spans[i][1]
                token_data["text"] = v["tokens"][i]
                token_data["prefix"] = "B" if i == 0 else "I" # if this is the first token in a mention
                token_data["realis"] = v["realis"] if "realis" in v else "Questionable"
                token_data["type_full"] = v["type_full"]
                token_data["type"] = v["type"]
                token_data["subtype"] = v["subtype"]
                token_data["mention"] = "EVENT"
                token_data["gold_anno_key_T"] = v["key"]

                tokens_with_labels.append(token_data)

                # add tokens in tree structure for faster search (we have a lot of tokens that will look for a match here!)
                if not token_data["begin"] in tokens_with_labels_spantree:
                    tokens_with_labels_spantree[token_data["begin"]] = {}
                tokens_with_labels_spantree[token_data["begin"]][token_data["end"]] = token_data

        return tokens_with_labels, tokens_with_labels_spantree

    #{'file_name': 'NYT_ENG_20130723.0097.txt', 'file_id': 'NYT_ENG_20130723.0097', u'sentences': [{u'tokens': [u'<DOC\xa0id="NYT_ENG_20130723.0097"\xa0type="story"\xa0from_file="/newswire/daily_process/nyt/english/source_data/20130723/20130723,59da641ef8c30ab5cee38ee978886813.xml">', u'<HEADLINE>', u'Drug', u'lab', u'</HEADLINE>', u'<TEXT>', u'<P>', u'EDNOTES', u':', u'-LRB-', u'FOR', u'USE', u'BY', u'NEW', u'YORK', u'TIMES', u'NEWS', u'SERVICE', u'CLIENTS', u'-RRB-', u'</P>', u'<P>', u'scandal', u'</P>', u'<P>', u'system', u'</P>', u'<P>', u'upheld', u'\xa9', u'2013', u'The', u'New', u'York', u'Times', u'</P>', u'<P>', u'The', u'state', u"'s", u'highest', u'court', u'has', u'ruled', u'that', u'judges', u'have', u'the', u'author\xc2ity', u'to', u'stay', u'the', u'sentences', u'of', u'so-called', u'Dookhan', u'defendants', u'who', u'are', u'seeking', u'new', u'trials', u',', u'upholding', u'the', u'legal', u'framework', u'created', u'to', u'review', u'hundreds', u'of', u'cases', u'tied', u'to', u'the', u'state', u'drug', u'lab', u'scandal', u'.'], u'lemmas': [u'<doc\xa0id="nyt_eng_20130723.0097"\xa0type="story"\xa0from_file="/newswire/daily_process/nyt/english/source_data/20130723/20130723,59da641ef8c30ab5cee38ee978886813.xml">', u'<HEADLINE>', u'Drug', u'lab', u'</headline>', u'<text>', u'<p>', u'ednote', u':', u'-lrb-', u'for', u'use', u'by', u'new', u'YORK', u'TIMES', u'NEWS', u'SERVICE', u'client', u'-rrb-', u'</p>', u'<p>', u'scandal', u'</p>', u'<p>', u'system', u'</p>', u'<p>', u'uphold', u'\xa9', u'2013', u'the', u'New', u'York', u'Times', u'</P>', u'<P>', u'the', u'state', u"'s", u'highest', u'court', u'have', u'rule', u'that', u'judge', u'have', u'the', u'author\xe2ity', u'to', u'stay', u'the', u'sentence', u'of', u'so-called', u'Dookhan', u'defendant', u'who', u'be', u'seek', u'new', u'trial', u',', u'uphold', u'the', u'legal', u'framework', u'create', u'to', u'review', u'hundred', u'of', u'case', u'tie', u'to', u'the', u'state', u'drug', u'lab', u'scandal', u'.'], u'pos': [u'NN', u'NNP', u'NNP', u'NN', u'NN', u'NN', u'NN', u'NNS', u':', u'-LRB-', u'IN', u'NN', u'IN', u'JJ', u'NNP', u'NNP', u'NNP', u'NNP', u'NNS', u'-RRB-', u'NN', u'NN', u'NN', u'NN', u'NN', u'NN', u'NN', u'NN', u'VBD', u'CD', u'CD', u'DT', u'NNP', u'NNP', u'NNP', u'NNP', u'NNP', u'DT', u'NN', u'POS', u'JJS', u'NN', u'VBZ', u'VBN', u'IN', u'NNS', u'VBP', u'DT', u'NN', u'TO', u'VB', u'DT', u'NNS', u'IN', u'JJ', u'NNP', u'NNS', u'WP', u'VBP', u'VBG', u'JJ', u'NNS', u',', u'VBG', u'DT', u'JJ', u'NN', u'VBN', u'TO', u'VB', u'NNS', u'IN', u'NNS', u'VBN', u'TO', u'DT', u'NN', u'NN', u'NN', u'NN', u'.'], u'char_offsets': [[0, 160], [161, 171], [172, 176], [177, 180], [181, 192], [193, 199], [200, 203], [204, 211], [211, 212], [213, 214], [214, 217], [218, 221], [222, 224], [225, 228], [229, 233], [234, 239], [240, 244], [245, 252], [253, 260], [260, 261], [262, 266], [267, 270], [271, 278], [279, 283], [284, 287], [288, 294], [295, 299], [300, 303], [304, 310], [311, 312], [313, 317], [318, 321], [322, 325], [326, 330], [331, 336], [337, 341], [342, 345], [346, 349], [350, 355], [355, 357], [358, 365], [366, 371], [372, 375], [376, 381], [382, 386], [387, 393], [394, 398], [399, 402], [403, 413], [414, 416], [417, 421], [422, 425], [426, 435], [436, 438], [439, 448], [449, 456], [457, 467], [468, 471], [472, 475], [476, 483], [484, 487], [488, 494], [494, 495], [496, 505], [506, 509], [510, 515], [516, 525], [526, 533], [534, 536], [537, 543], [544, 552], [553, 555], [556, 561], [562, 566], [567, 569], [570, 573], [574, 579], [580, 584], [585, 588], [589, 596], [596, 597]]}]}
    @staticmethod
    def tacdata_to_json_events2014(dir_src_txt, dir_ann_nugget_brat, parser, clear_txt=True, dir_out_clear=""):

        data = []
        data_dir_source = dir_src_txt # os.path.join(data_dir, "source")
        data_dir_annotation = dir_ann_nugget_brat # os.path.join(data_dir, "annotation")

        regex_get_doc_id = re.compile("id=\"(.*?)\"")
        for fname in os.listdir(data_dir_source):
            print "...processing %s" % (fname)
            if not fname.endswith(".txt") and not fname.endswith(".xml"):
                print "skipping"
                continue


            file_id = fname.replace(".txt", "") if fname.endswith(".txt") else fname.replace(".xml", "")

            file_name_source = os.path.join(data_dir_source, fname)
            file_name_annoation = os.path.join(data_dir_annotation, fname.replace(".txt", ".ann") if fname.endswith(".txt") else fname.replace(".xml", ".ann"))

            # read raw text
            # f = codecs.open(file_name_source, mode='rb', encoding='utf-8')
            # source_text = f.read()

            # read the text and clear tag lines but keep indexing
            source_text = ""
            f = codecs.open(file_name_source, mode='rb', encoding='utf-8')

            line_idx = 0
            for line in f:
                if line_idx == 0:
                    m = regex_get_doc_id.search(line)
                    if m:
                        file_id = m.group(1)
                        print "doc_id:%s" % file_id

                line_idx += 1

                if line.startswith("<") and not (line.startswith("<img")
                                                 # or line.startswith("<quote")
                                                 or line.startswith("<a")):
                    line = re.sub("[^\s]", " ", line)
                elif clear_txt:
                    # replace images
                    re_img_op = re.compile("(<img.*?>)")
                    repl_img_str = " <IMG> "

                    line_out = clear_matches_with_pos_preserve(line, re_img_op, repl_img_str)

                    # replace full hyperlinks
                    re_url_start_op = re.compile("(<a .*?/>)")
                    repl_str = " <URL> "
                    line_out = clear_matches_with_pos_preserve(line_out, re_url_start_op, repl_str)

                    # replace open url tag
                    re_url_full_op = re.compile("(<a .*?>)")
                    repl_str = " "
                    line_out = clear_matches_with_pos_preserve(line_out, re_url_full_op, repl_str)

                    # replace closing a tags
                    line_out = line_out.replace("</a>", "    ")

                    if len(line_out) == len(line):
                        line = line_out
                    else:
                        print ("Input line and cleared line does not match:\n")
                        print ("line__in:\"%s\"" % line)
                        print ("line_out:\"%s\"" % line_out)

                source_text += line

            if len(dir_out_clear) > 0:
                f_out_name = os.path.join(dir_out_clear, fname+".clear_txt")
                f_out = codecs.open(f_out_name, 'wb', encoding='utf-8')
                f_out.write(source_text)
                f_out.close()
                print "Written clear file:%s" % f_out_name

            # Parse the data
            data_source_parse = parser.parse_doc(source_text)
            print data_source_parse

            tokens_ann = []
            tokens_ann_tree = {}
            if os.path.isfile(file_name_annoation):
                read_coref_anno = True
                data_anno_nuggets, data_anno_coref = Tac2016_EventNuggets_DataUtilities.tacdata_load_annotations_events_from_brat_format(file_name_annoation, read_coref=read_coref_anno)
                tokens_ann, tokens_ann_tree = Tac2016_EventNuggets_DataUtilities.get_tokens_with_annotations(data_anno_nuggets)
                # print tokens_ann
                # print tokens_ann_tree
                data_source_parse["anno_nuggets"] = data_anno_nuggets
                data_source_parse["anno_coref"] = data_anno_coref
            else:
                print "%s - annotation file not found!" % file_name_annoation

            data_source_parse["file_id"] = file_id
            data_source_parse["file_name"] = fname

            # print tokens_ann_tree

            for sent_data in data_source_parse["sentences"]:
                # print "======================================================"
                # print sent_data
                labels_event = []
                labels_realis = []
                labels_type_full = []
                gold_anno_key_t = []
                labels_type = []
                labels_subtype = []
                for offset in sent_data["char_offsets"]:
                    lbl_event = "O"
                    lbl_realis = "O"
                    lbl_type_full = "O"
                    lbl_type = "O"
                    lbl_subtype = "O"
                    item_gold_anno_key_T = "O"

                    if offset[0] in tokens_ann_tree:
                        if offset[1] in tokens_ann_tree[offset[0]]:
                            ann = tokens_ann_tree[offset[0]][offset[1]]
                            # ann["prefix"]
                            # ann["realis"]
                            # ann["type_full"]
                            # ann["type"]
                            # ann["subtype"]

                            lbl_event = ann["prefix"]+"-"+ann["mention"]
                            lbl_realis = ann["prefix"]+"-"+ann["realis"]
                            lbl_type_full = ann["prefix"]+"-"+ann["type_full"]
                            item_gold_anno_key_T = ann["gold_anno_key_T"]
                            lbl_type = ann["type"]
                            lbl_subtype = ann["subtype"]

                    labels_event.append(lbl_event)
                    labels_realis.append(lbl_realis)
                    labels_type_full.append(lbl_type_full)
                    gold_anno_key_t.append(item_gold_anno_key_T)
                    labels_type.append(lbl_type)
                    labels_subtype.append(lbl_subtype)

                sent_data["labels_event"] = labels_event
                sent_data["labels_realis"] = labels_realis
                sent_data["labels_type_full"] = labels_type_full
                sent_data["gold_anno_key_T"] = gold_anno_key_t
                sent_data["labels_type"] = labels_type
                sent_data["labels_subtype"] = labels_subtype

            # print sent_data["labels_event"]
            # print sent_data["labels_type_full"]
            # print sent_data["tokens"]
            # print data_source_parse


            data.append(data_source_parse)
            #return data # debug
        return data

    @staticmethod
    def extract_event_nuggets(data_meta, include_event_sent_meta=False, include_doc_sentences_meta=False, use_per_doc_event_indexing=True):
        """
        Extracts event nuggets from data
        :param data_meta: Metadata in format as loaded from the json files produced by this class
        :param use_per_doc_event_indexing: If event indexing should be done per document - this is mainly used for debug purposes
        :return: Extracted event nuggets per document
        """
        docs_with_events = []

        # metadata fields:
        # tokens
        # deps_basic
        # lemmas
        # deps_cc
        # pos
        # parse
        # file_name
        # file_id
        # char_offsets
        # labels_realis
        # labels_type_full
        # labels_event

        curr_doc_id = ""
        event_id = 0

        curr_doc_data = {}

        for meta_id, data_item in enumerate(data_meta):
            try:
                doc_id = data_item["file_id"]
                if doc_id != curr_doc_id:
                    if curr_doc_id != "":
                        docs_with_events.append(curr_doc_data)

                    # start new doc data export
                    curr_doc_id = doc_id

                    curr_doc_data = {}
                    curr_doc_data["doc_id"] = curr_doc_id
                    curr_doc_data["file_id"] = curr_doc_id
                    curr_doc_data["file_name"] = data_item["file_name"]
                    curr_doc_data["sentences"] = []
                    if use_per_doc_event_indexing:
                        event_id = 0

                    curr_doc_data["event_nuggets"] = []

                if include_doc_sentences_meta:
                    curr_doc_data["sentences"].append(data_item)

                lbl_event_bevent = u"B-EVENT"
                lbl_event_ievent = u"I-EVENT"

                lbl_bevent_cnt = sum([1 for xx in data_item["labels_event"] if xx == lbl_event_bevent])
                # print "lbl_bevent_cnt:%s" % lbl_bevent_cnt
                lbl_ievent_cnt = sum([1 for xx in data_item["labels_event"] if xx == lbl_event_ievent])
                # print "lbl_ievent_cnt:%s" % lbl_ievent_cnt

                curr_event_token_i = 0
                sent_len = len(data_item["tokens"])
                for i in range(0, sent_len):
                    if data_item["labels_event"][i] == lbl_event_bevent:

                        event_id += 1
                        curr_event_token_i = i
                        curr_event_nugget = {}
                        curr_event_nugget["event_id"] = "E%s" % event_id

                        curr_event_nugget["gold_anno_key_T"] = data_item["gold_anno_key_T"][i] if "gold_anno_key_T" in data_item else "-"
                        # max search for I-EVENT for longer span events
                        max_span_tokens_search = 5  # In the data there is no span longer the 4 - we set 5 for convenience.
                        # Howerver longer would span would be an error.
                        last_ievent_idx = 0
                        if lbl_ievent_cnt > 0:
                            for j in range(i + 1, min(i + max_span_tokens_search, sent_len - 1)):
                                if data_item["labels_event"][j] == lbl_event_ievent:
                                    last_ievent_idx = j
                                    lbl_ievent_cnt -= 1

                                elif data_item["labels_event"] == lbl_event_bevent:
                                    break

                        if last_ievent_idx == 0:
                            last_ievent_idx = curr_event_token_i

                        nugget_span = (100000, 0)
                        nugget_txt = ""

                        curr_event_nugget["tokens"] = []
                        curr_event_nugget["char_offsets"] = []
                        curr_event_nugget["tokens_idx"] = []
                        range_curr = range(curr_event_token_i, last_ievent_idx + 1)

                        # print range_curr
                        for k in range_curr:
                            token = data_item["tokens"][k]
                            curr_event_nugget["tokens"].append(token)
                            curr_event_nugget["tokens_idx"].append(k)

                            char_offset = data_item["char_offsets"][k]
                            curr_event_nugget["char_offsets"].append(char_offset)

                            prev_token_span_to = nugget_span[1]
                            # add spaces if tokens are not connected to each other "I"<s>"eat"<s>"ham"
                            if prev_token_span_to > 0 and (char_offset[1] - prev_token_span_to - 1) > 0:
                                nugget_txt += " " * (char_offset[0] - prev_token_span_to)

                            nugget_txt += token

                            nugget_span = (min(nugget_span[0], char_offset[0]), char_offset[1])

                        curr_event_nugget["text"] = nugget_txt
                        curr_event_nugget["span"] = nugget_span
                        curr_event_nugget["realis"] = data_item["labels_realis"][i][2:]  # Assign the value of B-EVENT
                        curr_event_nugget["type_full"] = data_item["labels_type_full"][i][
                                                         2:]  # Assign the value of B-EVENT
                        # Confidence score
                        curr_event_nugget["span_confidence"] = 1.0
                        curr_event_nugget["realis_confidence"] = 1.0
                        curr_event_nugget["type_confidence"] = 1.0

                        curr_event_nugget["sent_id"] = -1
                        if include_doc_sentences_meta:
                            curr_event_nugget["sent_id"] = len(curr_doc_data["sentences"]) - 1

                        if include_event_sent_meta:
                            curr_event_nugget["sent_meta"] = data_item

                        curr_event_nugget["meta_id"] = meta_id
                        # print "E%s - %s" %(event_id,curr_event_nugget)

                        curr_doc_data["event_nuggets"].append(curr_event_nugget)
            except Exception as ex:
                logging.error("Error for item:\n%s"%str(data_item))
                raise ex

        # add the last document
        docs_with_events.append(curr_doc_data)

        return docs_with_events

    @staticmethod
    def save_to_output_tbf(file_name, docs_with_nuggets, system_name):
        f = codecs.open(file_name, mode='wb', encoding='utf-8')
        for doc in docs_with_nuggets:
            output_str = Tac2016_EventNuggets_DataUtilities.generate_output_file_content_tbf(doc, system_name)
            f.write(output_str)

        f.close()

    @staticmethod
    def save_to_output_tbf_filtertypes(file_name, docs_with_nuggets, system_name, allowed_types):
        f = codecs.open(file_name, mode='wb', encoding='utf-8')
        for doc in docs_with_nuggets:
            output_str = Tac2016_EventNuggets_DataUtilities.generate_output_file_content_tbf(doc,
                                                                                             system_name,
                                                                                             filter_types=True,
                                                                                             allowed_types=allowed_types)
            f.write(output_str)

        f.close()


    @staticmethod
    def generate_output_file_content_tbf(doc_with_nuggets, system_name, filter_types=False, allowed_types=None):
        def canonicalize_string(str):
            return "".join(c.lower() for c in str if c.isalnum())

        output_str = ""
        doc = doc_with_nuggets

        doc_id = doc["doc_id"]
        output_str += "#BeginOfDocument\t%s\n" % doc_id

        for event_nugget in doc["event_nuggets"]:
            event_id = event_nugget["event_id"]
            mention_text = event_nugget["text"]
            span = event_nugget["span"]
            realis = event_nugget["realis"]
            type_full = event_nugget["type_full"]

            # Confidence score
            span_confidence = event_nugget["span_confidence"]
            realis_confidence = event_nugget["realis_confidence"]
            type_confidence = event_nugget["type_confidence"]

            try:
                if filter_types and not canonicalize_string(event_nugget["type_full"]) in allowed_types:
                    continue
                output_str += "{system_name}\t{doc_id}\t{event_id}\t{span_begin},{span_end}\t{mention_text}\t{event_type}\t{realis}\t{event_span_confidence}\t{event_type_confidence}\t{realis_confidence}\n".format(
                    system_name=system_name,
                    doc_id=doc_id,
                    event_id=event_id,
                    span_begin=span[0],
                    span_end=span[1],
                    mention_text=mention_text,
                    event_type=type_full,
                    realis=realis,
                    event_span_confidence=span_confidence,
                    event_type_confidence=realis_confidence,
                    realis_confidence=type_confidence)
            except Exception as err:
                logging.error('Error exporting nugget from documents %s:%s' %(doc_id, event_nugget))


        output_str += "#EndOfDocument\n"

        # Coreference
        if "coreference_list" in doc:
            for coref in doc["coreference_list"]:
                coref_id = coref["coref_id"]
                coref_events_keys = coref["events_keys"]
                coref_events_keys_str = string.join(coref_events_keys, ",")

                output_str += "@Coreference\t{coref_id}\t{events_keys}\n".format(coref_id=coref_id,
                                                                                 events_keys=coref_events_keys_str)

        return output_str


    @staticmethod
    def get_data_idx_for_field(data_meta,
                                    field_name,
                                    field_vocab_dict,
                                    unknown_word):

        res = []
        for data_item in data_meta:
            new_item = [field_vocab_dict[x] if x in field_vocab_dict else field_vocab_dict[unknown_word] for x in data_item[field_name]]
            res.append(new_item)

        return res

    @staticmethod
    def get_left_right_dependency_labels_for_data_item(
                                        data_item,
                                         field_name,
                                         field_vocab_dict,
                                         unknown_lbl,
                                         zero_deps_lbl,
                                         field_sent_tokens):
        left_dict = {}  # dependency data for preds
        right_dict = {}  # dependency data for subjects
        res_left = []
        res_right = []
        for dep_item in data_item[field_name]:
            dep_label = dep_item[0]
            left_idx = dep_item[1]
            right_idx = dep_item[2]

            if left_idx in left_dict:
                left_dict[left_idx].append(dep_label)
            else:
                left_dict[left_idx] = [dep_label]

            if right_idx in right_dict:
                right_dict[right_idx].append(dep_label)
            else:
                right_dict[right_idx] = [dep_label]

        for i in range(0, len(data_item[field_sent_tokens])):
            curr_token_deps_left = []
            # Left deps
            if i in left_dict:
                deps_lbl_idxs = [field_vocab_dict[x] if x in field_vocab_dict else field_vocab_dict[unknown_lbl] for x
                                 in left_dict[i]]
                curr_token_deps_left = deps_lbl_idxs
            else:
                curr_token_deps_left = [field_vocab_dict[zero_deps_lbl]]

            res_left.append(curr_token_deps_left)

            # Right deps
            curr_token_deps_right = []
            if i in right_dict:
                deps_lbl_idxs = [field_vocab_dict[x] if x in field_vocab_dict else field_vocab_dict[unknown_lbl] for x
                                 in right_dict[i]]
                curr_token_deps_right = deps_lbl_idxs
            else:
                curr_token_deps_right = [field_vocab_dict[zero_deps_lbl]]

            res_right.append(curr_token_deps_right)

        return res_left, res_right

    @staticmethod
    def get_left_right_dependency_tokens_for_data_item(
            data_meta,
            field_name="deps_basic",
            field_sent_tokens="tokens",
            return_full_deps=False):

        left_dict = {}  # dependency data for preds
        right_dict = {}  # dependency data for subjects
        res_left = []
        res_right = []
        res_full = []

        for dep_item in data_meta[field_name]:
            dep_label = dep_item[0]
            left_idx = dep_item[1]
            right_idx = dep_item[2]

            left_item_dep = (dep_label, data_meta[field_sent_tokens][right_idx])
            if left_idx in left_dict:
                left_dict[left_idx].append(left_item_dep)
            else:
                left_dict[left_idx] = [left_item_dep]

            right_item_dep = (dep_label, data_meta[field_sent_tokens][left_idx])
            if right_idx in right_dict:
                right_dict[right_idx].append(right_item_dep)
            else:
                right_dict[right_idx] = [right_item_dep]

        for i in range(0, len(data_meta[field_sent_tokens])):
            # Left deps
            curr_token_deps_left = [x for x in left_dict[i]] if i in left_dict else []
            res_left.append(curr_token_deps_left)

            # Right deps
            curr_token_deps_right = [x for x in right_dict[i]] if i in right_dict else []
            res_right.append(curr_token_deps_right)

            # Full deps
            curr_token_deps_full = []
            curr_token_deps_full.extend(curr_token_deps_left)
            curr_token_deps_full.extend(curr_token_deps_right)
            res_full.append(curr_token_deps_full)
        if return_full_deps:
            return res_left, res_right, res_full
        else:
            return res_left, res_right

    @staticmethod
    def get_left_right_data_token_idx_for_dep_tokens(data_meta,
                                                      items_input_seq,
                                                      deps_field_name='deps_cc', # deps_cc, deps_basic
                                                      field_sent_tokens='tokens',
                                                      return_token_ids_in_sentence=True,
                                                      dep_depth=1,
                                                      include_curr_token=True,
                                                      include_zero_deps_token=False,
                                                      zero_deps_token_idx=0,  # this is the unknown token from the vocab
                                                      ):

        res_left_all = []
        res_right_all = []

        res_left_all_insent_ids = []
        res_right_all_insent_ids = []

        for item_id, data_item in enumerate(data_meta):
            left_dict = {}  # dependency data for preds
            right_dict = {}  # dependency data for subjects
            res_left = []
            res_right = []

            res_left_insent_ids = []
            res_right_insent_ids = []

            # collect tree edges
            for dep_item in data_item[deps_field_name]:
                dep_label = dep_item[0] # not used currently
                left_idx = dep_item[1]
                right_idx = dep_item[2]

                if left_idx in left_dict:
                    left_dict[left_idx].append(right_idx)
                else:
                    left_dict[left_idx] = [right_idx]

                if right_idx in right_dict:
                    right_dict[right_idx].append(left_idx)
                else:
                    right_dict[right_idx] = [left_idx]

            # retrieve dependencies of each token
            for i in range(0, len(data_item[field_sent_tokens])):
                curr_token_deps_left = []
                curr_token_deps_left_insent_idx = []
                if include_curr_token:
                    curr_token_deps_left.append(items_input_seq[item_id].x[i])
                    curr_token_deps_left_insent_idx.append(i)

                # Left deps
                if i in left_dict:
                    deps_token_idxs = [items_input_seq[item_id].x[x] for x in left_dict[i]]  # get the word tokens
                    curr_token_deps_left.extend(deps_token_idxs)
                    curr_token_deps_left_insent_idx.extend([x for x in left_dict[i]])
                else:
                    if include_zero_deps_token:
                        curr_token_deps_left.extend([zero_deps_token_idx])

                res_left.append(curr_token_deps_left)
                res_left_insent_ids.append(curr_token_deps_left_insent_idx)

                # Right deps
                curr_token_deps_right = []
                curr_token_deps_right_insent_idx = []
                if include_curr_token:
                    curr_token_deps_right.append(items_input_seq[item_id].x[i])
                    curr_token_deps_right_insent_idx.append(i)

                if i in right_dict:
                    deps_token_idxs = [items_input_seq[item_id].x[x] for x in right_dict[i]]
                    curr_token_deps_right.extend(deps_token_idxs)
                    curr_token_deps_right_insent_idx.extend([x for x in right_dict[i]])
                else:
                    if include_zero_deps_token:
                        curr_token_deps_right.extend([zero_deps_token_idx])

                res_right.append(curr_token_deps_right)
                res_right_insent_ids.append(curr_token_deps_right_insent_idx)

            res_left_all.append(res_left)
            res_right_all.append(res_right)

            res_left_all_insent_ids.append(res_left_insent_ids)
            res_right_all_insent_ids.append(res_right_insent_ids)

        return res_left_all, res_right_all, res_left_all_insent_ids, res_right_all_insent_ids

    @staticmethod
    def get_left_right_data_idx_for_deps(data_meta,
                                             field_name,
                                             field_vocab_dict,
                                             unknown_lbl,
                                             zero_deps_lbl,
                                             field_sent_tokens
                                             ):

            res_left_all = []
            res_right_all = []
            for data_item in data_meta:
                left_dict = {}  # dependency data for preds
                right_dict = {}  # dependency data for subjects
                res_left = []
                res_right = []
                for dep_item in data_item[field_name]:
                    dep_label = dep_item[0]
                    left_idx = dep_item[1]
                    right_idx = dep_item[2]

                    if left_idx in left_dict:
                        left_dict[left_idx].append(dep_label)
                    else:
                        left_dict[left_idx] = [dep_label]

                    if right_idx in right_dict:
                        right_dict[right_idx].append(dep_label)
                    else:
                        right_dict[right_idx] = [dep_label]

                for i in range(0, len(data_item[field_sent_tokens])):
                    curr_token_deps_left = []
                    # Left deps
                    if i in left_dict:
                        deps_lbl_idxs = [field_vocab_dict[x] if x in field_vocab_dict else field_vocab_dict[unknown_lbl]
                                         for x in left_dict[i]]
                        curr_token_deps_left = deps_lbl_idxs
                    else:
                        curr_token_deps_left = [field_vocab_dict[zero_deps_lbl]]

                    res_left.append(curr_token_deps_left)

                    # Right deps
                    curr_token_deps_right = []
                    if i in right_dict:
                        deps_lbl_idxs = [field_vocab_dict[x] if x in field_vocab_dict else field_vocab_dict[unknown_lbl]
                                         for x
                                         in right_dict[i]]
                        curr_token_deps_right = deps_lbl_idxs
                    else:
                        curr_token_deps_right = [field_vocab_dict[zero_deps_lbl]]

                    res_right.append(curr_token_deps_right)

                res_left_all.append(res_left)
                res_right_all.append(res_right)

            return res_left_all, res_right_all


    # @staticmethod
    # def export_to_submission_format_tbf_txt():
    #     export_txt = ""
    #
    #     # metadata fields:
    #     # tokens
    #     # deps_basic
    #     # lemmas
    #     # deps_cc
    #     # pos
    #     # parse
    #     # file_name
    #     # file_id
    #     # char_offsets
    #     # labels_realis
    #     # labels_type_full
    #     # labels_event
    #
    #
    #     return export_txt

###############################
#Sample usage:

#2014 tac data - train - python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training/source" -dir_ann_nugget_brat:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training/annotation" -output_file:"data_tac2014_train.json" -dir_src_out_clear:"_clear_src"
#2014 tac data - eval - python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/eval/source" -dir_ann_nugget_brat:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/eval/annotation" -output_file:"data_tac2014_eval.json" -dir_src_out_clear:"_clear_src"
#2015 tac data - train - python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/training/source" -dir_ann_nugget_brat:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/training/bratNuggetAnn" -output_file:"data_tac2015_train.json" -dir_src_out_clear:"_clear_src"
#2015 tac data - eval - python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/eval/source" -dir_ann_nugget_brat:"/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/eval/bratNuggetAnn" -output_file:"data_tac2015_eval.json" -dir_src_out_clear:"_clear_src"
###############################

if __name__ == '__main__':
    # input_dirs = "/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training"
    #
    # dir_list = []
    # if len(sys.argv)>0:
    #     input_dirs = sys.argv[1]
    #
    #     dir_list = [x.strip() for x in input_dirs.split(";")]
    #     print('Text input_dirs:\n')
    #     for dir_name in dir_list:
    #         print dir_name
    #
    # else:
    #     print('Error: missing input file parameter')
    #     quit()

    dir_src_txt = ""
    dir_src_txt = CommonUtilities.get_param_value("dir_src_txt", sys.argv, dir_src_txt)
    dir_src_txt_dirlist = dir_src_txt.split(";")

    has_error = False

    dir_src_out_clear = ""
    dir_src_out_clear = CommonUtilities.get_param_value("dir_src_out_clear", sys.argv, dir_src_out_clear)
    dir_src_out_clear_dirlist = dir_src_out_clear.split(";")
    print "dir_src_out_clear:%s" % dir_src_out_clear
    if len(dir_src_out_clear_dirlist)>0 and len(dir_src_txt_dirlist)<>len(dir_src_out_clear_dirlist):
        has_error = True
        print('Error: Number of dirs in dir_src_out_clear_dirlist(%s) does not match the number of dirs in dir_src_txt_dirlist(%s)' %(len(dir_src_out_clear_dirlist), len(dir_src_txt_dirlist)))

    dir_ann_hopper_brat = ""
    dir_ann_hopper_brat = CommonUtilities.get_param_value("dir_ann_hopper_brat", sys.argv, dir_ann_hopper_brat)
    dir_ann_hopper_brat_dirlist = dir_ann_hopper_brat.split(";")
    print "dir_ann_hopper_brat:%s" % dir_ann_hopper_brat

    dir_ann_nugget_brat = ""
    dir_ann_nugget_brat = CommonUtilities.get_param_value("dir_ann_nugget_brat", sys.argv, dir_ann_nugget_brat)
    dir_ann_nugget_brat_dirlist = dir_ann_nugget_brat.split(";")
    print "dir_ann_nugget_brat:%s" % dir_ann_nugget_brat


    if len(dir_src_txt) == 0:
        has_error = True
        print('Error: missing dir_src_txt file parameter')

    if (len(dir_ann_nugget_brat) == 0) and (len(dir_ann_hopper_brat) == 0):
        has_error = True
        print('Error: either  dir_ann_nugget_brat or dir_ann_hopper_brat should be specified')

    if has_error:
        quit()

    command = "convert_to_json_2014"
    command = CommonUtilities.get_param_value("cmd", sys.argv, command)

    output_file = "output_file"
    output_file = CommonUtilities.get_param_value("output_file", sys.argv, output_file)

    coreNlpPath = "/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets-1/corenlp/stanford-corenlp-full-2015-12-09/*"
    coreNlpPath = CommonUtilities.get_param_value("coreNlpPath", sys.argv, coreNlpPath)
    if(command=="convert_to_json_2014"):
        data_format = "tac2014"
        print "Data format:%s" % data_format

        parse_mode = "parse" # "pos"
        parser = CoreNLP(parse_mode, corenlp_jars=[coreNlpPath])

        start = time.time()

        data = []
        for dir_idx in range(len(dir_src_txt_dirlist)):
            curr_dir_src_txt = dir_src_txt_dirlist[dir_idx]
            curr_dir_ann_nugget_brat = dir_ann_nugget_brat_dirlist[dir_idx]

            # dir to output the preprocessed fields
            dir_src_out_clear = ""
            if dir_idx < len(dir_src_out_clear_dirlist):
                dir_src_out_clear = dir_src_out_clear_dirlist[dir_idx]
                if not os.path.exists(dir_src_out_clear):
                    os.makedirs(dir_src_out_clear)
                    print "%s created!" % dir_src_out_clear

            json_data = []
            if data_format == "tac2014":
                print "parsing"
                print "curr_dir_src_txt: %s" % curr_dir_src_txt
                print "curr_dir_ann_nugget_brat: %s" % curr_dir_ann_nugget_brat
                json_data = Tac2016_EventNuggets_DataUtilities.tacdata_to_json_events2014(curr_dir_src_txt, curr_dir_ann_nugget_brat, parser=parser, dir_out_clear=dir_src_out_clear)

            data.extend(json_data)
        end = time.time()
        print("Done in %s s"%(end - start))

        print len(data)
        Tac2016_EventNuggets_DataUtilities.save_data_to_json_file(data, output_json_file=output_file)
        print ("Data exported to %s" % output_file)
    else:
        print "No command param specified: use -cmd:convert_to_json_2014 or -cmd:convert_to_json_2015 "




