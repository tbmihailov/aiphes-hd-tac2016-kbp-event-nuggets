
from sklearn.metrics import confusion_matrix
import numpy as np

def pad(seq, pad_value, to_size):
    pad_seq = []
    if len(seq) > to_size:
        pad_seq = seq[:to_size]
    else:
        pad_seq = seq[:] + [pad_value]*(to_size - len(seq))

    return pad_seq


def print_acc_from_conf_matrix(conf_matrix, classes_dict):
    res = {}

    for lblid in range(conf_matrix.shape[0]):
        overl_cnt = np.sum(conf_matrix[lblid])
        true_cnt = conf_matrix[lblid][lblid]
        lbl_acc = float(true_cnt)/float(overl_cnt) if overl_cnt > 0 else 1
        lbl_name = classes_dict[lblid+1] if lblid+1 in classes_dict else "None"
        print "%s (%s) accuracy = %s"%(lbl_name.ljust(10), lblid, lbl_acc)

def get_acc_from_conf_matrix(conf_matrix, classes_dict):
    res = {}
    for lblid in range(conf_matrix.shape[0]):
        overl_cnt = np.sum(conf_matrix[lblid])
        true_cnt = conf_matrix[lblid][lblid]
        lbl_acc = float(true_cnt)/float(overl_cnt) if overl_cnt > 0 else 1
        lbl_name = classes_dict[lblid+1] if lblid+1 in classes_dict else "None"

        res[lbl_name] = lbl_acc

    return res

def get_prec_rec_fscore_acc_from_conf_matrix(conf_matrix, classes_dict):
    """
    Calculates prec, rec, f-score(MicroAvg), Acc from a confusion matrix
    :param conf_matrix: Cnfusion matrix
    :param classes_dict: Classes dictionary having matrix idx as key and class name as value
    :return: Dictionary with key class name and value Tuple (prec, rec, f-score(MicroAvg), Acc)
    """
    res = {}
    for lblid in range(conf_matrix.shape[0]):
        all_pos_cnt = np.sum(conf_matrix, axis=1)[lblid]  # np.sum(conf_matrix[lblid])
        pred_pos_cnt = np.sum(conf_matrix, axis=0)[lblid]
        tp_cnt = conf_matrix[lblid][lblid]
        fp_cnt = pred_pos_cnt - tp_cnt
        fn_cnt = all_pos_cnt - tp_cnt

        prec = float(tp_cnt)/(float(tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 1)
        recall = float(tp_cnt)/(float(tp_cnt + fn_cnt) if (tp_cnt + fn_cnt) > 0 else 1)

        acc = prec
        f_score = (2*(prec * recall)/(prec + recall)) if (prec+recall>0) else 0

        lbl_name = classes_dict[lblid+1] if lblid+1 in classes_dict else "None"

        res[lbl_name] = (prec, recall, f_score, acc)

    return res



def prepare_batch_data(data, pad_value=0):
    batch_data_seqlens = np.asarray([len(a.x) for a in data])
    max_len = max(batch_data_seqlens)

#     batch_data_padded_x = [pad(a.x.tolist(), pad_value, max_len) for a in data]
#     batch_data_padded_y = [pad(a.y.tolist(), pad_value, max_len) for a in data]

    batch_data_padded_x = np.asarray([pad(a.x, pad_value, max_len) for a in data])
    batch_data_padded_y = np.asarray([pad(a.y, pad_value, max_len) for a in data])
    # batch_data_padded_x_embedd = [[pad(a.x.tolist(), pad_value, max_len)] for a in batch_data]
    # batch_data_padded_y_mask = [[1 if item>0 else 0 for item in pad(a.y.tolist(), pad_value, max_len)] for a in batch_data]

    return batch_data_padded_x, batch_data_padded_y, batch_data_seqlens

def pad_and_get_mask(data, pad_value=0):
    # data is of rank 3: [batch_size, number_tokens, number_deps]

    # number tokens
    batch_data_seqlens = [len(a) for a in data]
    max_len_seq = max(batch_data_seqlens)

    # max number of deps
    max_sub_length = 0
    for item in data:
        # print "item:%s"%item
        for token in item:
            if len(token) > max_sub_length:
                max_sub_length = len(token)

    # Build the input
    data_padded = []
    mask = []
    data_padded_seqlens = []
    for item in data:
        item_padded = []
        item_mask = []
        item_seqlens = []
        for token_deps in item:

            token_deps_padded = token_deps + [pad_value]*(max_sub_length-len(token_deps))
            item_padded.append(token_deps_padded)

            token_deps_mask = [1]*len(token_deps)+[0]*(max_sub_length-len(token_deps))
            item_mask.append(token_deps_mask)
            item_seqlens.append(max(1, sum(token_deps_mask)))

        item_padded = item_padded + (max_len_seq-len(item_padded))*[[pad_value]*max_sub_length]
        data_padded.append(item_padded)

        item_mask = item_mask + (max_len_seq - len(item_mask)) * [[pad_value] * max_sub_length]
        mask.append(item_mask)

        item_seqlens = item_seqlens + (max_len_seq - len(item_mask)) * [1]
        data_padded_seqlens.append(item_seqlens)


    return data_padded, data_padded_seqlens, mask
