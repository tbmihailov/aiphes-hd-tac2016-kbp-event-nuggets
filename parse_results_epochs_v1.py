import re
import codecs
import sys
import string

file_name = sys.argv[1]


def extract_epochs_accuracies(file_name):
    f = codecs.open(file_name, mode='rb', encoding='utf-8')
    epoch_results = {}

    re_get_epoch_id = "Epoch ([\d]+):"
    re_get_curr_score_id = "Curr\sf-score:([\d]+\.*[\d]*)"

    epoch_id = None
    acc = None
    for line in f:
        # print line
        m = re.search(re_get_epoch_id, line)
        if m:
            epoch_id_new = int(m.group(1))
            if epoch_id is not None:
                epoch_results[epoch_id] = acc

            epoch_id = epoch_id_new
            acc = None

        m1 = re.search(re_get_curr_score_id, line)
        if m1:
            acc = float(m1.group(1))

    if not epoch_id in epoch_results:
        epoch_results[epoch_id] = acc

    f.close()

    epoch_ids = [k for k, v in epoch_results.items()]
    max_epoch_id = max(epoch_ids)
    if max_epoch_id is None:
        max_epoch_id=0
    indexed_epochs = [None] * (max_epoch_id + 1)
    for i in range(0, len(indexed_epochs)):
        if i in epoch_results:
            indexed_epochs[i] = epoch_results[i]

    return indexed_epochs


results = extract_epochs_accuracies(file_name)

import numpy as np

print "%s\t%s\t%s" % (file_name, np.argmax(np.asarray(results)), string.join(["%s" % x for x in results], "\t"))


