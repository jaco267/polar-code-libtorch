#pragma once
#include <torch/torch.h>
#include <iostream>
using namespace std;
class PolarEncoder{
  public: 
    PolarEncoder(//* constructor 
      const torch::Tensor& frozen_pos,
      int n,  const torch::Tensor& G
    );   
    PolarEncoder(){};//default polarenc constructor
    torch::Tensor forward(const torch::Tensor& u) const;
  private:
    int n_;
    int k_; 
    torch::Tensor frozen_pos_;
    torch::Tensor info_pos_;
    torch::Tensor G_;  
};