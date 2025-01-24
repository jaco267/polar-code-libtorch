#pragma once
#include <torch/torch.h>
torch::Tensor get_Kern_frozen_bits(
  int n,int f_num,torch::Tensor kern,
  torch::Tensor & frozen_pos);