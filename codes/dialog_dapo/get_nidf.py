import json
import codecs
import math
import tqdm
total=49930

projectdir="../../"
ifile=projectdir+"datasets/dialog/rawtext_dialog_ngram.json"
ofile=projectdir+"datasets/dialog/dialog_rawtext_nidf.json"
with codecs.open(ifile,'r',"utf8") as f:
    ngram=json.load(f)

ngram2=[{},{},{}]
ngram2minmax=[[100.0,-100.0],[100.0,-100.0],[100.0,-100.0]]
for i in range(3):
    for j in ngram[i]:
        idf=math.log(total/float(ngram[i][j]),10)
        if idf<ngram2minmax[i][0]:
            ngram2minmax[i][0]=idf
        if idf>ngram2minmax[i][1]:
            ngram2minmax[i][1]=idf
        ngram2[i][j]=idf

ngram3 = [{}, {}, {}]
for i in range(3):
    for j in ngram2[i]:
        nidf=(ngram2[i][j]-ngram2minmax[i][0])/(ngram2minmax[i][1]-ngram2minmax[i][0])
        ngram3[i][j]=nidf

with codecs.open(ofile,"w","utf8") as f:
    json.dump(ngram3,f)
