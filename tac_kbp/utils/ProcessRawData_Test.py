from nltk.tokenize import RegexpTokenizer

import codecs

fname = "/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets/data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training/source/0f03cc5a508d630c6c8c8c61396e31a9.tkn.txt"
text = "<a>Good muffins cost $3.88\nin New York.</a>  <p> Please buy me\ntwo of them.\n\nThanks.</p>"
f = codecs.open(fname, encoding='utf-8')
text = f.read()

import ConfigParser
import numpy as np
from stanford_corenlp_pywrapper import CoreNLP

coreNlpPath = "/home/mihaylov/Programming/TAC2016/tac2016-kbp-event-nuggets/corenlp/stanford-corenlp-full-2015-12-09/*"

print "Load proc"
proc = CoreNLP("pos", corenlp_jars=[coreNlpPath])
print "start"
res = proc.parse_doc(text)

for sent in res["sentences"]:
    print sent