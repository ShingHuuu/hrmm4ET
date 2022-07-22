"""Pytorch modules."""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import constant

from typing import Dict, Optional, Tuple
from hrm_wrapper import BoxTensor, log1mexp
from hrm_wrapper import CenterBoxTensor
from hrm_wrapper import CenterSigmoidBoxTensor
import os


euler_gamma = 0.6


def _compute_gumbel_min_max(
  box1: BoxTensor,
  box2: BoxTensor,
  gumbel_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns min and max points."""
  min_point = torch.stack([box1.z, box2.z])
  min_point = torch.max(
    gumbel_beta * torch.logsumexp(min_point / gumbel_beta, 0),
    torch.max(min_point, 0)[0])

  max_point = torch.stack([box1.Z, box2.Z])
  max_point = torch.min(
    -gumbel_beta * torch.logsumexp(-max_point / gumbel_beta, 0),
    torch.min(max_point, 0)[0])
  return min_point, max_point


def _compute_hard_min_max(
  box1: BoxTensor,
  box2: BoxTensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns min and max points."""
  min_point = torch.max(box1.z, box2.z)
  max_point = torch.min(box1.Z, box2.Z)
  return min_point, max_point


class LinearProjection(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               bias: bool = True):
    super(LinearProjection, self).__init__()
    self.linear = nn.Linear(input_dim, output_dim, bias=bias)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    outputs = self.linear(inputs)
    return outputs


class HighwayNetwork(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               n_layers: int,
               activation: Optional[nn.Module] = None):
    super(HighwayNetwork, self).__init__()
    self.n_layers = n_layers
    self.nonlinear = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    self.gate = nn.ModuleList(
      [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
    for layer in self.gate:
      layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
    self.final_linear_layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.ReLU() if activation is None else activation
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    for layer_idx in range(self.n_layers):
      gate_values = self.sigmoid(self.gate[layer_idx](inputs))
      nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
      inputs = gate_values * nonlinear + (1. - gate_values) * inputs
    return self.final_linear_layer(inputs)


class TypeSelfAttentionLayer(nn.Module):
  def __init__(self,
               scale: float = 1.0,
               attn_dropout: float = 0.0):
    super(TypeSelfAttentionLayer, self).__init__()
    self.scale = scale
    self.dropout = nn.Dropout(attn_dropout)

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = torch.matmul(q, k.transpose(1, 2)) / self.scale
    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)
    attn = self.dropout(F.softmax(attn, dim=-1))
    output = torch.matmul(attn, v)
    return output, attn


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim: int, answer_num_l1: int, answer_num_l2: int):
    super(SimpleDecoder, self).__init__()
    self.answer_num_l1 = answer_num_l1
    self.answer_num_l2 = answer_num_l2
    self.linear_l1 = nn.Linear(output_dim, answer_num_l1, bias=False)
    self.linear_l2 = nn.Linear(output_dim, answer_num_l2, bias=False)

  def forward(self,
              inputs: torch.Tensor) -> torch.Tensor:
    output_embed = self.linear(inputs)
    return output_embed


class BoxDecoder(nn.Module):

  box_types = {
    'BoxTensor': BoxTensor,
    'CenterSigmoidBoxTensor': CenterSigmoidBoxTensor
  }

  def __init__(self,
               args: argparse.Namespace,
               answer_num_l1: int,
               answer_num_l2: int,
               embedding_dim: int,
               box_type: str,
               batchsize_num: int,
               goal_data: str,
               padding_idx: Optional[int] = None,
               max_norm: Optional[float] = None,
               norm_type: float = 2.,
               scale_grad_by_freq: bool = False,
               sparse: bool = False,
               _weight: Optional[torch.Tensor] = None,
               init_interval_delta: float = 0.5,
               init_interval_center: float = 0.01,
               inv_softplus_temp: float = 1.,
               softplus_scale: float = 1.,
               n_negatives: int = 0,
               neg_temp: float = 0.,
               box_offset: float = 0.5,
               pretrained_box: Optional[torch.Tensor] = None,
               use_gumbel_baysian: bool = True,
               gumbel_beta: float = 1.0,
               offset_scale_l1: torch.tensor = torch.randn(5,1),
               offset_scale_l2: torch.tensor = torch.randn(5,1)
               ):
    super(BoxDecoder, self).__init__()

    self.box_embedding_dim = embedding_dim
    self.answer_num_l1 = answer_num_l1
    self.answer_num_l2 = answer_num_l2
    self.args = args
    self.word2id_ = constant.load_vocab_dict(constant.TYPE_FILES[self.args.goal])
    self.word2id_l1_, self.word2id_l2_ = constant.load_vocab_dict_hierachy(constant.TYPE_FILES[self.args.goal])
    self.id2word_l1_ = {v: k for k, v in self.word2id_l1_.items()}
    self.id2word_l2_ = {v: k for k, v in self.word2id_l2_.items()}
    self.key_list = list(self.word2id_.keys())
    self.id2word_ = {v: k for k, v in self.word2id_.items()}
    self.box_type = box_type
    self.type_density_l1 = nn.Parameter(torch.randn(1,len(self.id2word_l1_)), requires_grad=True)
    self.type_density_l2 = nn.Parameter(torch.randn(1,len(self.id2word_l2_)), requires_grad=True)
    self.linear_density = nn.Linear(self.box_embedding_dim,1)
    self.goal_data = goal_data

    try:
      self.box = self.box_types[box_type]
    except KeyError as ke:
      raise ValueError("Invalid box type {}".format(box_type)) from ke

    self.box_offset = box_offset  # Used for constant tensor
    self.offset_scale_l1 = offset_scale_l1
    self.offset_scale_l2 = offset_scale_l2
    self.init_interval_delta = init_interval_delta
    self.init_interval_center = init_interval_center
    self.inv_softplus_temp = inv_softplus_temp
    self.softplus_scale = softplus_scale
    self.n_negatives = n_negatives
    self.neg_temp = neg_temp
    self.use_gumbel_baysian = use_gumbel_baysian
    self.gumbel_beta = gumbel_beta
    self.box_embeddings = nn.Embedding(self.answer_num_l1 + self.answer_num_l2,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)

    self.box_embeddings_l1 = nn.Embedding(self.answer_num_l1,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)

    self.box_embeddings_l2 = nn.Embedding(self.answer_num_l2,
                                       embedding_dim * 2,
                                       padding_idx=padding_idx,
                                       max_norm=max_norm,
                                       norm_type=norm_type,
                                       scale_grad_by_freq=scale_grad_by_freq,
                                       sparse=sparse,
                                       _weight=_weight)

    if pretrained_box is not None:
      print('Init box emb with pretrained boxes.')
      print(self.box_embeddings.weight)
      self.box_embeddings.weight = nn.Parameter(pretrained_box)

  def init_weights(self):
    print('before', self.box_embeddings.weight)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., :self.box_embedding_dim],
      -self.init_interval_center, self.init_interval_center)
    torch.nn.init.uniform_(
      self.box_embeddings.weight[..., self.box_embedding_dim:],
      self.init_interval_delta, self.init_interval_delta)
    print('after', self.box_embeddings.weight)

  # 传入两个density
  def log_soft_volume(self,
    z: torch.Tensor,
    Z: torch.Tensor,
    mention_density: torch.Tensor,
    type_density: torch.Tensor,
    box_name: str,
    temp: float = 1.,
    scale: float = 1.,
    gumbel_beta: float = 0.
    ) -> torch.Tensor:
    eps = torch.finfo(z.dtype).tiny  # type: ignore

    # print('vol_type',vol_type)
    if isinstance(scale, float):
      s = torch.tensor(scale)
    else:
      s = scale

    if box_name == 'mention':
        density_scale = mention_density.t()
    else:
        density_scale = (type_density.repeat(mention_density.size()[0],1) + mention_density)/2

    if gumbel_beta <= 0.:
        return (torch.sum(
        torch.log(F.softplus((Z - z), beta=temp).clamp_min(eps)),
        dim=-1) + torch.log(s)
              )  # need this eps to that the derivative of log does not blow
    else:
          #   mention的密度 值直接传进来，相交部分的密度值 ，用加和除以2
        aaa = torch.sum(
        torch.log(
          F.softplus(Z - z - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(
            eps)),
          dim=-1) + torch.log(s) + density_scale
        # aaa += density_scale
        return aaa



  def forward(
    self,
    mc_box: torch.Tensor,
    targets_l1: Optional[torch.Tensor] = None,
    targets_l2: Optional[torch.Tensor] = None,
    is_training: bool = True,
    batch_num: Optional[int] = None,
    offset_scale_l1: torch.Tensor=torch.randn(5,1),
    offset_scale_l2: torch.Tensor=torch.randn(5,1)
  ) -> Tuple[torch.Tensor, None]:

    inputs_l1 = torch.arange(0,
                          self.answer_num_l1,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)

    inputs_l2 = torch.arange(0,
                          self.answer_num_l2,
                          dtype=torch.int64,
                          device=self.box_embeddings.weight.device)

    emb_l1 = self.box_embeddings_l1(inputs_l1)  # num types x 2*box_embedding_dim
    emb_l2 = self.box_embeddings_l2(inputs_l2)


    if self.box_type == 'ConstantBoxTensor':
      print("first_box_offset",self.box_offset.device)
      type_box_l1 = self.box.from_split(emb_l1, self.box_offset)
      type_box_l2 = self.box.from_split(emb_l2,self.box_offset)
    else:
      type_box_l1 = self.box.from_split(emb_l1, offset_scale_l1)
      type_box_l2 = self.box.from_split(emb_l2,offset_scale_l2)

    # Get intersection
    batch_size = mc_box.data.size()[0]

    tmp_density_l1 = torch.sigmoid(self.type_density_l1)
    tmp_density_l2 = torch.sigmoid(self.type_density_l2)
    mention_density = torch.sigmoid(self.linear_density(mc_box.z))+torch.sigmoid(self.linear_density(mc_box.Z))
    mention_density = mention_density + torch.ones_like(mention_density)


    # min calculation

    min_point_l1 = torch.stack(
    [(mc_box.z.unsqueeze(1)).expand(-1, len(self.id2word_l1_), -1),
     (type_box_l1.z_type.unsqueeze(0)).expand(batch_size, -1, -1)])

    min_point_l2 = torch.stack(
    [mc_box.z.unsqueeze(1).expand(-1, len(self.id2word_l2_), -1),
     type_box_l2.z_type.unsqueeze(0).expand(batch_size, -1, -1)])

    min_point_l1 = torch.max(
    self.gumbel_beta * torch.logsumexp(min_point_l1 / self.gumbel_beta, 0),
    torch.max(min_point_l1, 0)[0])

    min_point_l2 = torch.max(
    self.gumbel_beta * torch.logsumexp(min_point_l2 / self.gumbel_beta, 0),
    torch.max(min_point_l2, 0)[0])

    # max calculation
    max_point_l1 = torch.stack([
    mc_box.Z.unsqueeze(1).expand(-1, len(self.id2word_l1_), -1),
    type_box_l1.Z_type.unsqueeze(0).expand(batch_size, -1, -1)])

    max_point_l2 = torch.stack([
    mc_box.Z.unsqueeze(1).expand(-1, len(self.id2word_l2_), -1),
    type_box_l2.Z_type.unsqueeze(0).expand(batch_size, -1, -1)])

    max_point_l1 = torch.min(
    -self.gumbel_beta * torch.logsumexp(-max_point_l1/ self.gumbel_beta, 0),
    torch.min(max_point_l1, 0)[0])

    max_point_l2 = torch.min(
    -self.gumbel_beta * torch.logsumexp(-max_point_l2/ self.gumbel_beta, 0),
    torch.min(max_point_l2, 0)[0])

    # 中间态结果
    # torch.save(min_point_l1,"./bbn_min_point_l1.pt")
    # torch.save(max_point_l1,"./bbn_max_point_l1.pt")
    # torch.save(min_point_l2,"./bbn_min_point_l2.pt")
    # torch.save(max_point_l2,"./bbn_max_point_l2.pt")
    # os.exit()

    vol1_l1 = self.log_soft_volume(min_point_l1,
                                max_point_l1,
                                mention_density,
                                tmp_density_l1,
                                'type',
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta,
                                )

    vol1_l2 = self.log_soft_volume(min_point_l2,
                                max_point_l2,
                                mention_density,
                                tmp_density_l2,
                                'type',
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta,
                                )

    # Compute the volume of the mention&context box
    vol2 = self.log_soft_volume(mc_box.z,
                                mc_box.Z,
                                mention_density,
                                mention_density,
                                'mention',
                                temp=self.inv_softplus_temp,
                                scale=self.softplus_scale,
                                gumbel_beta=self.gumbel_beta,
                                )

    # Returns log probs
    log_probs_l1 = vol1_l1 - vol2.t()
    # log_probs_l1 = F.logsigmoid(log_probs_l1)

    log_probs_l2 = vol1_l2 - vol2.t()
    # log_probs_l2 = F.logsigmoid(log_probs_l2)

    # Clip values > 1. for numerical stability.
    if (log_probs_l1 > 0.0).any():
      print("WARNING: Clipping log probability since it's grater than 0.")
      log_probs_l1[log_probs_l1 > 0.0] = 0.0

    # Clip values > 1. for numerical stability.
    if (log_probs_l2 > 0.0).any():
      print("WARNING: Clipping log probability since it's grater than 0.[l1]")
      log_probs_l2[log_probs_l2 > 0.0] = 0.0

    if is_training and targets_l1 is not None and self.n_negatives > 0:
      return log_probs_l1, None, targets_l1, targets_l2, log_probs_l2
    elif is_training and targets_l1 is not None and self.n_negatives <= 0:
      return log_probs_l1, None, targets_l1, targets_l2, log_probs_l2
    else:
      return log_probs_l1, None, None, None, log_probs_l2