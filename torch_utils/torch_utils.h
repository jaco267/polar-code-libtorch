#pragma once  
torch::Tensor setdiff1d(const torch::Tensor& a, const torch::Tensor& b);
torch::Tensor binary_repr(int x, int bits);