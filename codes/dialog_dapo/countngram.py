import nltk
import json
import codecs
from collections import Counter
from tqdm import *

projectdir="../../"
ifile=projectdir+"datasets/dialog/rawtext_dialog.json"
ofile=projectdir+"/datasets/dialog/rawtext_dialog_ngram.json"

def sortdict(dict):
    return sorted(dict.items(), key=lambda d: d[1], reverse=True)

# return the list of n-gram in a utterance
def utttongram(n,utt):
    utt=utt.lower()
    sens=nltk.sent_tokenize(utt)
    words=[]
    for sent in sens:
        words.append(nltk.word_tokenize(sent))
    ngramlist=[]
    for tokenized_words in words:
        for i in range(len(tokenized_words)-n+1):
            n_gram=" " .join(tokenized_words[i:i+n])
            ngramlist.append(n_gram)
    return ngramlist # # #


grams=[[],[],[]]


with codecs.open(ifile,"r","utf8") as f:
    rawdata=json.load(f)


for example in tqdm(rawdata):
    for utt in example['text']:
        for i in range(1,4):
            grams[i-1]+=utttongram(i,utt)

inputdata={"1gram":dict(Counter(grams[0])),"2gram":dict(Counter(grams[1])),"3gram":dict(Counter(grams[2]))}

with codecs.open(ofile,"w","utf-8") as f:
    json.dump(inputdata,f)

