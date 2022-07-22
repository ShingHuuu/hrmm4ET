

import math
import torch
import torch.nn.functional as F

from scipy import special
from torch import Tensor
from typing import Tuple, Union, Type, TypeVar
import argparse

tanh_eps = 1e-20
_log1mexp_switch = math.log(0.5)
parser = argparse.ArgumentParser()
device_ = torch.device("cuda")

"""
update data
"""

goal_data = 'bbn'
print("check goal data",goal_data)

def log1mexp(x: torch.Tensor,
             split_point=_log1mexp_switch,
             exp_zero_eps=1e-7) -> torch.Tensor:
  logexpm1_switch = x > split_point
  Z = torch.zeros_like(x)
  logexpm1 = torch.log((-torch.expm1(x[logexpm1_switch])).clamp_min(1e-38))

  logexpm1_bw = torch.log(-torch.expm1(x[logexpm1_switch]) + exp_zero_eps)
  Z[logexpm1_switch] = logexpm1.detach() + (logexpm1_bw - logexpm1_bw.detach())
  Z[~logexpm1_switch] = torch.log1p(-torch.exp(x[~logexpm1_switch]))

  return Z

def log1pexp(x: torch.Tensor) -> torch.Tensor:
  Z = torch.zeros_like(x)
  zone1 = (x <= 18.)
  zone2 = (x > 18.) * (x < 33.3)  # And operator using *
  zone3 = (x >= 33.3)
  Z[zone1] = torch.log1p(torch.exp(x[zone1]))
  Z[zone2] = x[zone2] + torch.exp(-(x[zone2]))
  Z[zone3] = x[zone3]

  return Z


def _box_shape_ok(t: Tensor) -> bool:
  if len(t.shape) < 2:
    return False
  else:
    if t.size(-2) != 2:
      return False

    return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
  return "Shape of {} has to be {} but is {}".format(tensor_name,
                             expected_shape,
                             tuple(actual_shape))


class ExpEi(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    dev = input.device
    with torch.no_grad():
      x = special.exp1(input.detach().cpu()).to(dev)
      input.to(dev)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = grad_output*(-torch.exp(-input)/input)
    return grad_input


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):

  def __init__(self,data: Tensor) -> None:
    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(_shape_error_str('data', '(**,2,num_dims)', data.shape))
    super().__init__()

  def __repr__(self):
    return 'box_tensor_wrapper(' + self.data.__repr__() + ')'

  @property
  def z(self) -> Tensor:
    """Lower left coordinate as Tensor"""

    return self.data[..., 0, :]

  @property
  def Z(self) -> Tensor:
    """Top right coordinate as Tensor"""

    return self.data[..., 1, :]

  @property
  def centre(self) -> Tensor:
    """Centre coordinate as Tensor"""

    return (self.z + self.Z)/2

  @classmethod
  def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor, offset_scale: Tensor) -> TBoxTensor:

    if z.shape != Z.shape:
      raise ValueError(
        "Shape of z and Z should be same but is {} and {}".format(
          z.shape, Z.shape))
    box_val: Tensor = torch.stack((z, Z), -2)

    return cls(box_val, offset_scale = offset_scale)

  @classmethod
  def from_split(cls: Type[TBoxTensor], t: Tensor,offset_scale,
           dim: int = -1) -> TBoxTensor:
    len_dim = t.size(dim)

    if len_dim % 2 != 0:
      raise ValueError(
        "dim has to be even to split on it but is {}".format(
          t.size(dim)))
    split_point = int(len_dim / 2)
    z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point)), dtype=torch.int64, device=t.device))

    Z = t.index_select(
      dim,
      torch.tensor(
        list(range(split_point, len_dim)),
        dtype=torch.int64,
        device=t.device))

    return cls.from_zZ(z, Z,offset_scale)
class CenterSigmoidBoxTensor(BoxTensor):

  def __init__(self, data: Tensor, offset_scale: Tensor = torch.randn(5,1)) -> None:


    if _box_shape_ok(data):
      self.data = data
    else:
      raise ValueError(
        _shape_error_str('data', '(**,2,num_dims)', data.shape))
    super(CenterSigmoidBoxTensor, self).__init__(data)
    self.goal = goal_data

    self.offset_scale = offset_scale.to(data.device)



  @property
  def center(self) -> Tensor:
    return self.data[..., 0, :]

  @property
  def z(self) -> Tensor:
    z = self.data[..., 0, :] \
      - torch.nn.functional.softplus(self.data[..., 1, :]*self.offset_scale , beta=10.)
    return torch.sigmoid(z)

  @property
  def Z(self) -> Tensor:
    Z = self.data[..., 0, :] \
      + torch.nn.functional.softplus(self.data[..., 1, :]*self.offset_scale, beta=10.)
    return torch.sigmoid(Z)


  @property
  def z_type(self) -> Tensor:
    z = self.data[..., 0, :] \
      - torch.nn.functional.softplus(self.data[..., 1, :]*self.offset_scale, beta=10.)

    return torch.sigmoid(z)

  @property
  def Z_type(self) -> Tensor:

    Z = self.data[..., 0, :] \
      + torch.nn.functional.softplus(self.data[..., 1, :]*self.offset_scale, beta=10.)
    return torch.sigmoid(Z)


class CenterBoxTensor(BoxTensor):

  @property
  def center(self) -> Tensor:
    return self.data[..., 0, :]

  @property
  def z(self) -> Tensor:
    #return self.data[..., 0, :] - torch.sigmoid(self.data[..., 1, :])
    return self.data[..., 0, :] \
         - torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)

  @property
  def Z(self) -> Tensor:
    #return self.data[..., 0, :] + torch.sigmoid(self.data[..., 1, :])
    return self.data[..., 0, :] \
         + torch.nn.functional.softplus(self.data[..., 1, :], beta=10.)
