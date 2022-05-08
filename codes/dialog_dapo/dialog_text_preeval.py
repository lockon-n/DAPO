import csv
import json
import codecs
import random
import nltk
import tqdm
from collections import Counter
import copy

projectdir="../../"
ifile=projectdir+"datasets/dialog/rawtext_dialog.json"
ifile2=projectdir+"datasets/dialog/dialog_rawtext_nidf.json"
ofile=projectdir+"datasets/dialog/rawtext_dialog_score.csv"
maxturns=10


shortened_text=[] # list of {'id':1-1,'text':text}


with codecs.open(ifile,"r","utf8") as f:
    rawdata=json.load(f)

with codecs.open(ifile2,"r","utf8") as f:
    globalngraminfo = json.load(f)

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
def constructUO(dialog):
    t=copy.deepcopy(dialog['text'])
    random.shuffle(t)
    return {"id":dialog['id']+"_UO","text":t}
def constructUI(dialog):
    text=copy.deepcopy(dialog['text'])
    lentext=len(text)
    removeid=random.randint(0,lentext-1)
    insertid=random.randint(0,lentext-2)
    rutt = text[removeid]
    del text[removeid]
    text.insert(insertid, rutt)
    return {"id":dialog['id']+'_UI',"text":text}
def constructUR(dialog,lib):
    id_=int(dialog['id'][0])
    text=copy.deepcopy(dialog['text'])
    replaceturnid=random.randint(0,len(text)-1)
    selecteddiaid=random.randint(0,49929)
    while selecteddiaid==id_:
        selecteddiaid = random.randint(0, 49929)
    selectturn=random.choice(lib[selecteddiaid])
    return {'id':dialog['id']+"_UR","text":text[:replaceturnid]+[selectturn]+text[replaceturnid+1:]}
def _get_ngram_score_(ngram_dialog,ngram_global):
    total=0
    score=0
    for i in ngram_dialog:
        total+=ngram_dialog[i]
    for ngram_ in ngram_dialog:
        tf=ngram_dialog[ngram_]/float(total)
        nidf=ngram_global[ngram_]
        score+=tf*nidf
    return score
def getngramscore(dialog,ngraminfo):
    ngram=[[],[],[]]
    text=dialog['text']
    for utt in text:
        ngram[0]+=utttongram(1,utt)
        ngram[1]+=utttongram(2,utt)
        ngram[2]+=utttongram(3,utt)
    refinedngram=[dict(Counter(i)) for i in ngram]
    return [_get_ngram_score_(refinedngram[0],ngraminfo[0]),
            _get_ngram_score_(refinedngram[1],ngraminfo[1]),
            _get_ngram_score_(refinedngram[2],ngraminfo[2])]


lib={}

for example in tqdm.tqdm(rawdata):
    id=example['dialog_id']
    text=example['text']
    lib[id] = text

    if len(text)<=maxturns:
        example={"id":str(id)+"_0","text":text}
        shortened_text.append(example)
    else:
        for i in range(len(text)-maxturns+1):
            x=text[i:i+maxturns]
            example={"id":str(id)+"_"+str(i),"text":x}
            shortened_text.append(example)

data=[]
for dialog in tqdm.tqdm(shortened_text):
    scores=getngramscore(dialog,globalngraminfo)
    a=" ".join(dialog['text'])
    turns=len(dialog['text'])
    words=len(nltk.word_tokenize(a))
    originone=(dialog['id'],a,scores[0],scores[1],scores[2],turns,words)
    UOdialog=constructUO(dialog)
    UIdialog=constructUI(dialog)
    URdialog=constructUR(dialog,lib)
    b=" ".join(UOdialog['text'])
    UOone = (UOdialog['id'], b , -1,-1,-1,turns,words)
    c=" ".join(UIdialog['text'])
    UIone = (UIdialog['id'], c, -1,-1,-1,turns,words)
    d=" ".join(URdialog['text'])
    URone = (URdialog['id'], d, -1,-1,-1,turns,len(nltk.word_tokenize(d)))
    data.append(originone)
    data.append(UOone)
    data.append(UIone)
    data.append(URone)




with codecs.open(ofile,"w","utf8") as f:
    writer=csv.writer(f)
    writer.writerows(data)