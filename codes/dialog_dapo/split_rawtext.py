import csv
import random
import codecs
import tqdm

projectdir="../../"
ifile=projectdir+"datasets/dialog/rawtext_dialog_score.csv"
ofile1=projectdir+"datasets/dialog/rawtext_dialog_score_train.csv"
ofile2=projectdir+"datasets/dialog/rawtext_dialog_score_valid.csv"

trainlist=[]
validlist=[]

def write_csv(data,filename):
    with codecs.open(filename,"w","utf8") as f:
        writer=csv.writer(f)
        writer.writerows(data)

with codecs.open(ifile,"r","utf8") as f:
    f_csv=csv.reader(f)
    for row in tqdm.tqdm(f_csv):
        if random.randint(1,10)==1:
            validlist.append(row)
        else:
            trainlist.append(row)
random.shuffle(trainlist)
random.shuffle(validlist)
write_csv(trainlist,ofile1)
write_csv(validlist,ofile2)
