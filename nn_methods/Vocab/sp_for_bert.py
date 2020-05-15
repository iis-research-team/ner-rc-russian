import os
import sys
import json
import nltk
import random
import sentencepiece as spm
from tensorflow.keras.utils import Progbar

regex_tokenizer = nltk.RegexpTokenizer("\w+")

def normalize_text(text):
  # lowercase text
  text = str(text).lower()
  # remove non-UTF
  text = text.encode("utf-8", "ignore").decode()
  # remove punktuation symbols
  # text = " ".join(regex_tokenizer.tokenize(text))
  return text

def count_lines(filename):
  count = 0
  with open(filename) as fi:
    for line in fi:
      count += 1
  return count


#RAW_DATA_FPATH = "eco_texts.txt" 
#PRC_DATA_FPATH = "proc_eco_texts.txt"
RAW_DATA_FPATH = "sci_texts.txt" 
PRC_DATA_FPATH = "proc_sci_texts.txt"


total_lines = count_lines(RAW_DATA_FPATH)
bar = Progbar(total_lines)

with open(RAW_DATA_FPATH,encoding="utf-8") as fi:
  with open(PRC_DATA_FPATH, "w",encoding="utf-8") as fo:
    for l in fi:
      fo.write(normalize_text(l)+"\n")
      bar.add(1)

MODEL_PREFIX = "tokenizer" #@param {type: "string"}
#VOC_SIZE = 32000 #@param {type:"integer"}
VOC_SIZE = 119547 #@param {type:"integer"}
SUBSAMPLE_SIZE = 10000000 #@param {type:"integer"}
NUM_PLACEHOLDERS = 10000 #@param {type:"integer"}

SPM_COMMAND = ('--input={} --model_prefix={} '
               '--vocab_size={} --input_sentence_size={} '
               '--shuffle_input_sentence=true ' 
               '--hard_vocab_limit=false ' # my add for sci_texts
               '--bos_id=-1 --eos_id=-1 --model_type=bpe').format(
               PRC_DATA_FPATH, MODEL_PREFIX, 
               VOC_SIZE - NUM_PLACEHOLDERS, SUBSAMPLE_SIZE)

spm.SentencePieceTrainer.Train(SPM_COMMAND)


def read_sentencepiece_vocab(filepath):
  voc = []
  with open(filepath, encoding='utf-8') as fi:
    for line in fi:
      voc.append(line.split("\t")[0])
  # skip the first unk token
  voc = voc[1:]
  return voc


snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))

def parse_sentencepiece_token(token):
    if token.startswith("▁"):
        return token[1:]
    else:
        return "##" + token
        
bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

ctrl_symbols = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab
bert_vocab += ["[UNUSED_{}]".format(i) for i in range(VOC_SIZE - len(bert_vocab))]
print(len(bert_vocab))

VOC_FNAME = "vocab.txt" #@param {type:"string"}

with open(VOC_FNAME, "w") as fo:
  for token in bert_vocab:
    fo.write(token+"\n")

