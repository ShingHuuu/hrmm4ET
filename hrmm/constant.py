import torch
import json
import os

from typing import Dict, Optional


CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"

ANSWER_NUM_DICT = { "onto": 89, "figer": 113, "bbn": 56}

# Specify paths here
BASE_PATH = "./"
FILE_ROOT = "../data/"
EXP_ROOT = "../model"
ONTOLOGY_DIR = "../data/ontology"
bert_base_path = os.path.join(EXP_ROOT,"bert_base_uncased")
bert_large_path = os.path.join(EXP_ROOT,"bert_large_uncased")

TYPE_FILES = {
    "onto": os.path.join(ONTOLOGY_DIR,  "ontonotes_types.txt"),
    "figer": os.path.join(ONTOLOGY_DIR, "figer_types.txt"),
    "bbn": os.path.join(ONTOLOGY_DIR, "bbn_types.txt")
}


def load_vocab_dict(
  vocab_file_name: str,
  vocab_max_size: Optional[int] = None,
  start_vocab_count: Optional[int] = None,
  common_vocab_file_name: Optional[str] = None
) -> Dict[str, int]:
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if common_vocab_file_name:
        print("==> adding common training set types")
        print("==> before:", len(text))
        with open(common_vocab_file_name, "r") as fc:
            common = [x.strip() for x in fc.readlines()]
        print("==> common:", len(common))
        text = list(set(text + common))
        print("==> after:", len(text))
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) +
                                          start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content

def load_vocab_dict_hierachy(
  vocab_file_name: str,
) -> Dict[str, int]:
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    text_l1 = [x for x in text if x == '/'.join(x.split('/')[:2])]
    text_l2 = text
    file_content_l1 = dict(zip(text_l1, range(0, len(text_l1))))
    file_content_l2 = dict(zip(text_l2, range(0, len(text_l2))))
  return file_content_l1, file_content_l2