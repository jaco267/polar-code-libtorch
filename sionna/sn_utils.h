#pragma once
#include <torch/torch.h>
#include <iostream>
using namespace std;  

torch::Tensor complex_normal(c10::ArrayRef<int64_t> shape);

torch::Tensor expand_to_rank(torch::Tensor tensor, int target_rank,int axis);