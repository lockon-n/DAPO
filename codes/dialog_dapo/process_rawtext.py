import json
import codecs


projectdir="../../"
ofile=projectdir+"datasets/dialog/rawtext_dialog.json"

global_count=0
count1=0
count2=0
count3=0
count4=0
global_text=[]

ifile1=projectdir+"datasets/dialog/blended_skill_talk/train.json"
ifile2=projectdir+"datasets/dialog/blended_skill_talk/test.json"
ifile3=projectdir+"datasets/dialog/blended_skill_talk/valid.json"
ifile123=[ifile1,ifile2,ifile3]
for ifile in ifile123:
    with codecs.open(ifile,"r","utf8") as f:
        rawdata1=json.load(f)
    for example in rawdata1:
        dialog=example['dialog']
        combined_text=[x[1] for x in dialog]
        global_text.append({'dialog_id':global_count,'text':combined_text})
        global_count += 1
        count1+=1



ifile4=projectdir+"datasets/dialog/convai2_personachat/train_none_original_no_cands.txt"
ifile5=projectdir+"datasets/dialog/convai2_personachat/valid_none_original_no_cands.txt"
ifile45=[ifile4,ifile5]
for ifile in ifile45:
    with codecs.open(ifile,"r","utf8") as f:
        lines=f.readlines()
    prev=-1
    dialog=[]
    for line in lines:
        linedata=line.split("\t")
        turn_id=int(linedata[0][0])
        utt1=linedata[0][2:]
        utt2=linedata[1][:-1]
        if turn_id>prev:
            dialog.append(utt1)
            dialog.append(utt2)
        else:
            combined_text=dialog
            global_text.append({'dialog_id': global_count, 'text': combined_text})
            global_count += 1
            count2 += 1
            dialog=[utt1,utt2]
        prev=turn_id
    combined_text=dialog
    global_text.append({'dialog_id': global_count, 'text': combined_text})
    global_count += 1
    count2+=1



ifile6=projectdir+"datasets/dialog/ijcnlp_dailydialog/dialogues_text.txt"
with codecs.open(ifile6,"r","utf8") as f:
    lines=f.readlines()
for line in lines:
    combined_text=[i for i in line.split('__eou__')][:-1]
    global_text.append({'dialog_id': global_count, 'text': combined_text})
    global_count += 1
    count3+=1


ifile7=projectdir+"datasets/dialog/topicalchat/train.json"
ifile8=projectdir+"datasets/dialog/topicalchat/test_freq.json"
ifile9=projectdir+"datasets/dialog/topicalchat/test_rare.json"
ifile10=projectdir+"datasets/dialog/topicalchat/valid_freq.json"
ifile11=projectdir+"datasets/dialog/topicalchat/valid_rare.json"
ifilerest=[ifile7,ifile8,ifile9,ifile10,ifile11]
for ifile in ifilerest:
    with codecs.open(ifile,"r","utf8") as f:
        data=json.load(f)
    for i in data:
        example=data[i]
        dialog=example["content"]
        combined_text=[x['message'] for x in dialog]
        global_text.append({'dialog_id': global_count, 'text': combined_text})
        global_count += 1
        count4+=1


print(count1,count2,count3,count4)
with codecs.open(ofile,"w","utf8") as f:
    json.dump(global_text,f)
