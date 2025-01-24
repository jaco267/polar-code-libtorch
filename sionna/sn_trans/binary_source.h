#pragma once
#include <iostream>
#include <torch/torch.h>
using namespace std;
class BinarySource{
public:
  BinarySource(torch::Dtype dtype,torch::Device device);
  // BinarySource() : dtype_(torch::kInt8), device_(torch::kCPU) {cout<<"binary source init"<<endl;} // Default constructor //dont use this  
  torch::Tensor forward(const vector<int64_t>& size);
  ~BinarySource(){};  
private:/* data */
  torch::Device device_ = torch::kCPU;
  torch::Dtype dtype_ = torch::kInt8; 
};


