import argparse
import glob
import json
import numpy as np
import torch
import os
from random import shuffle
from tqdm import tqdm
from typing import Any, Dict, Generator, Optional, Tuple, Union

import constant


def to_torch(
  batch: Dict,
  device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
  inputs_to_model = {k: v.to(device) for k, v in batch["inputs"].items()}
  targets = batch["targets"].to(device)
  targets_l1 = batch['targets_l1'].to(device)
  targets_l2 = batch['targets_l2'].to(device)
  return inputs_to_model, targets, targets_l1, targets_l2


def get_example(
  generator: Generator[
    Tuple[Any, Union[list, Any], Union[list, Any], Any, list], None, None],
  batch_size: int,
  max_len: int,
  eval_data: bool = False,
  tokenizer: Optional[object] = None,
  answer_num: int = 60000,
  answer_num_l1: int = 6000,
  answer_num_l2: int = 6000
) -> Generator[Dict[str, np.ndarray], None, None]:
  # [cur_stream elements]
  # 0: example id, 1: left context, 2: right context, 3: mention word, 4: gold category, 5:gold level1 category
  cur_stream = [None] * batch_size
  no_more_data = False
  while True:
    bsz = batch_size
    mention_length_limit = 10  # in words, not word pieces
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if no_more_data and bsz == 0:
      break
    ex_ids = np.zeros([bsz], np.object)
    targets = np.zeros([bsz, answer_num], np.float32)
    targets_l1 = np.zeros([bsz, answer_num_l1], np.float32)
    targets_l2 = np.zeros([bsz, answer_num_l2], np.float32)
    inputs = []
    sentence_len_wp = []
    for i in range(bsz):
      ex_ids[i] = cur_stream[i][0]
      left_seq = cur_stream[i][1]
      right_seq = cur_stream[i][2]
      mention_seq = cur_stream[i][3]
      if len(mention_seq) > mention_length_limit:
        mention_seq = mention_seq[:mention_length_limit]
      mention = ' '.join(mention_seq)
      context = ' '.join(left_seq + mention_seq + right_seq)
      len_after_tokenization = len(
        tokenizer.encode_plus(mention, context)["input_ids"])
      if len_after_tokenization > max_len:
        overflow_len = len_after_tokenization - max_len
        context = " ".join(left_seq + mention_seq + right_seq[:-overflow_len])
      inputs.append([mention, context])
      len_after_tokenization = len(
        tokenizer.encode_plus(mention, context)["input_ids"])
      sentence_len_wp.append(len_after_tokenization)
      # Gold categories
      for answer_ind in cur_stream[i][4]:
        targets[i, answer_ind] = 1.0
      for answer_id in cur_stream[i][5]:
        targets_l1[i,answer_id] = 1.0
      for answer_id2 in cur_stream[i][6]:
        targets_l2[i,answer_id2] = 1.0
    max_len_in_batch = max(sentence_len_wp)

    inputs = tokenizer.batch_encode_plus(
      inputs,
      add_special_tokens=True,
      max_length=min(max_len, max_len_in_batch),
      truncation_strategy="only_second",
      pad_to_max_length=True,
      return_tensors="pt",
      truncation=True
    )
    targets = torch.from_numpy(targets)
    targets_l1 = torch.from_numpy(targets_l1)
    targets_l2 = torch.from_numpy(targets_l2)
    feed_dict = {"ex_ids": ex_ids, "inputs": inputs, "targets": targets, "targets_l1":targets_l1, "targets_l2":targets_l2}

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class DatasetLoader(object):

  def __init__(self,
               filepattern: str,
               args: argparse.Namespace,
               tokenizer: object):
      self._all_shards = [file for file in glob.glob(filepattern)
                          if "allwikipp_wiki" not in file]
      shuffle(self._all_shards)
      print("Found %d shards at %s" % (len(self._all_shards), filepattern))
      self.word2id = constant.load_vocab_dict(constant.TYPE_FILES[args.goal])
      self.id2word =  {v: k for k, v in self.word2id.items()}
      self.tokenizer = tokenizer
      self.do_lower = args.do_lower
      self.context_window_size = args.context_window_size
      self.args = args
      self.word2id_l1, self.word2id_l2 = constant.load_vocab_dict_hierachy(constant.TYPE_FILES[args.goal])
      self.id2word_l1 = {v: k for k, v in self.word2id_l1.items()}
      self.id2word_l2 = {v: k for k, v in self.word2id_l2.items()}
  def _load_npz(self, path: str) -> np.ndarray:
      with open(path, 'rb') as f:
          data = np.load(f)
      return data



  def _load_shard(self, shard_name: str) -> zip:
      print("Loading {}".format(shard_name))
      with open(shard_name) as f:
        lines = [json.loads(line.strip()) for line in tqdm(f)]
        ex_ids = [line["ex_id"] for line in lines]
        mention_word = [
          line["word"].split() if not self.do_lower
          else line["word"].lower().split() for line in lines]
        left_seq = [
          line["left_context"][-self.context_window_size:] if not self.do_lower
          else [
                 w.lower() for w in line["left_context"]
               ][-self.context_window_size:] for line in lines]
        right_seq = [
          line["right_context"][:self.context_window_size] if not self.do_lower
          else [
                 w.lower() for w in line["right_context"]
               ][:self.context_window_size] for line in lines]

        y_categories = [line["y_category"] for line in lines]
        y_category_ids = []
        y_category_ids_l1 = []
        y_category_ids_l2 = []
        for iid, y_strs in enumerate(y_categories):
          y_category_ids.append(
            [self.word2id[x] for x in y_strs if x in self.word2id])
          y_category_ids_l1.append(
            [self.word2id_l1[x] for x in y_strs if x in self.word2id_l1])
          tmp_l2_list = []
          h2_tmp = [ii for ii in y_strs if len(ii.split('/'))>2]
          if (len(y_strs) == len(y_category_ids_l1[-1])):
            for xx in y_strs:
              tmp_l2_list.append(self.word2id_l2[xx])
          else:
            for xx in y_strs:
              if (xx != '/'.join(xx.split('/')[:2])):
                tmp_l2_list.append(self.word2id_l2[xx])
              else:
                for xxx in h2_tmp:
                  if  xx == '/'.join(xx.split('/')[:2]):
                    break
                else:
                  tmp_l2_list.append(self.word2id_l2[xxx])
                  print("targets", y_strs)
                  print("l2", [self.id2word_l2[xxx] for xxx in tmp_l2_list])
                  print("*******")

          y_category_ids_l2.append(tmp_l2_list)
        return zip(ex_ids, left_seq, right_seq, mention_word,  y_category_ids, y_category_ids_l1, y_category_ids_l2)

  def _get_sentence(self, epoch: int):
    for i in range(0, epoch):
      for shard in self._all_shards:
        ids = self._load_shard(shard)
        for current_ids in ids:
          yield current_ids

  def get_batch(
    self,
    batch_size: int,
    max_len: int,
    epoch: int,
    eval_data: bool = True
  ) -> Generator[Dict[str, np.ndarray], None, None]:
      return get_example(
        self._get_sentence(epoch),
        batch_size=batch_size,
        max_len=max_len,
        eval_data=eval_data,
        tokenizer=self.tokenizer,
        answer_num=constant.ANSWER_NUM_DICT[self.args.goal],
        answer_num_l1=len(self.word2id_l1),
        answer_num_l2=len(self.word2id))


class DatasetLoaderForEntEval(object):

  def __init__(self, data: Any, args: argparse.Namespace, tokenizer: object):
    self.data = data
    self.word2id = constant.load_vocab_dict(constant.TYPE_FILES[args.goal])
    self.word2id_l1, self.word2id_l2 = constant.load_vocab_dict_hierachy(constant.TYPE_FILES[args.goal])
    self.tokenizer = tokenizer
    self.do_lower = args.do_lower
    self.context_window_size = args.context_window_size
    self.args = args

  def _load_npz(self, path: str) -> np.ndarray:
    with open(path, "rb") as f:
      data = np.load(f)
    return data

  def _load_shard(self):
    lines = self.data
    ex_ids = [line["ex_id"] for line in lines]
    mention_word = [
    line["word"].split() if not self.do_lower
    else line["word"].lower().split() for line in lines]
    left_seq = [
    line["left_context"][-self.context_window_size:] if not self.do_lower
    else [
         w.lower() for w in line["left_context"]
       ][-self.context_window_size:] for line in lines]
    right_seq = [
    line["right_context"][:self.context_window_size] if not self.do_lower
    else [w.lower() for w in line["right_context"]][:self.context_window_size]
        for line in lines]
    y_categories = [line["y_category"] for line in lines]
    y_category_ids = []
    y_category_ids_l1 = []
    y_category_ids_l2 = []

    for iid, y_strs in enumerate(y_categories):
      y_category_ids.append(
        [self.word2id[x] for x in y_strs if x in self.word2id])
      y_category_ids_l1.append(
        [self.word2id_l1[x] for x in y_strs if x in self.word2id_l1])
      tmp_l2_list = []
      h2_tmp = [ii for ii in y_strs if len(ii.split('/')) > 2]
      if (len(y_strs) == len(y_category_ids_l1[-1])):
        for xx in y_strs:
          tmp_l2_list.append(self.word2id_l2[xx])
      else:
        for xx in y_strs:
          if (xx != '/'.join(xx.split('/')[:2])):
            tmp_l2_list.append(self.word2id_l2[xx])
          else:
            for xxx in h2_tmp:
              if xx == '/'.join(xx.split('/')[:2]):
                break
            else:
              tmp_l2_list.append(self.word2id_l2[xxx])
              print("targets", y_strs)
              print("l2", [self.id2word_l2[xxx] for xxx in tmp_l2_list])
              print("*******")

      y_category_ids_l2.append(tmp_l2_list)
    return zip(ex_ids, left_seq, right_seq, mention_word,  y_category_ids, y_category_ids_l1, y_category_ids_l2)

  def _get_sentence(self):
    ids = self._load_shard()
    for current_ids in ids:
      yield current_ids

  def get_batch(
    self,
    batch_size: int,
    max_len: int,
    epoch: int,
    eval_data: bool = True
  ) -> Generator[Dict[str, np.ndarray], None, None]:
      return get_example(
        self._get_sentence(),
        batch_size=batch_size,
        max_len=max_len,
        eval_data=eval_data,
        tokenizer=self.tokenizer,
        answer_num=constant.ANSWER_NUM_DICT[self.args.goal],
        answer_num_l1=len(self.word2id_l1),
        answer_num_l2=len(self.word2id_l2))