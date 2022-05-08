# Dialogue-adaptive Pre-training from Quality Estimation

This is the code for the paper **Dialogue-adaptive Pre-training from Quality Estimation**



<h3>1. Requirements</h3>

(Our experiment environment for reference)

Python 3.7+

Pytorch (1.0.0)

NLTK (3.4.5)



 <h3>2. Datasets</h3>

**2.1 Datasets for constructing pre-training corpus**

[DaliyDialog](http://yanran.li/files/ijcnlp_dailydialog.zip) ---->./datasets/dialog/ijcnlp_daillydialog/

[PERSONA-CHAT](https://dl.fbaipublicfiles.com/parlai/convai2/convai2_fix_723.tgz) ---->./datasets/dialog/convai2_personachat/

[Topical-Chat](https://github.com/alexa/Topical-Chat/tree/master/conversations) ---->./datasets/dialog/topicalchat/

[BlendedSkillTalk](http://parl.ai/downloads/blended_skill_talk/blended_skill_talk.tar.gz) ---->./datasets/dialog/blended_skill_talk/

After downloading these datasets, extract them to the corresponding directories.

**2.2 Datasets for Downstream Tasks**

[MuTual&MuTual$^{plus}$](https://github.com/Nealcly/MuTual/tree/master/data)

[DailyDialog&PERSONA-CHAT(Annotated)](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt?usp=sharing)

[FED](http://shikib.com/fed_data.json) 

We slightly pre-process the datasets so that they have a uniform format. The pre-processed data can be found in ./datasets/



<h3>3. Instructions</h3>

<h4>3.1 Construct Pre-training Corpus</h4>

**Get raw text of the dialogues**

```bash
python ./codes/dialog_dapo/process_rawtext.py
```

**Count the n-grams**

```bash
python ./codes/dialog_dapo/countngram.py
```

**Get n-NIDFs**

```bash
python ./codes/dialog_dapo/get_nidf.py
```

**Bulid the pre-training corpus**

```bash
python ./codes/dialog_dapo/dialog_text_preeval.py
```

**Split the pre-training corpus**

```bash
python ./codes/dialog_dapo/split_rawtext.py
```

**Move the pre-training corpus to the target directory**

```bash
mv ./datasets/dialog/rawtext_dialog_score_train.csv ./datasets/dialog_eval_pretrain/rawtext_pretrain/train.csv
mv ./datasets/dialog/rawtext_dialog_score_dev.csv ./datasets/dialog_eval_pretrain/rawtext_pretrain/dev.csv
```

<h4>3.2 Pre-training ELECTRA with DAPO and fine-tuning on downstream tasks</h4>

We provide the scripts used for pre-training and fine-tuning.

**Pre-training**

```bash
sh ./codes/dialog_dapo/scripts/pretrain_myptALL_3_NIDF.sh
```

**Fine-tuning**

```bash
sh ./codes/dialog_dapo/scripts/downstream_myptALL_3_NIDF.sh
```

The results can be found in ./results/electraDAPO_myptALL_3_NIDF/electraDAPO_myptALL_3_NIDF_downstream_log_results.txt.

