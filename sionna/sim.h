#pragma once
#include <torch/torch.h>
#include "awgn_model.h"
void sim_ber(System_Awgn_model* mc_fun,
  torch::Tensor ebno_dbs,int batch_size,
  int max_mc_iter,Tensor& ber,Tensor& bler);