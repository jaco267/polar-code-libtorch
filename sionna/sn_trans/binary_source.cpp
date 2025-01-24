#include <iostream>
#include <torch/torch.h>
#include "binary_source.h"
using namespace std;
BinarySource::BinarySource(torch::Dtype dtype,torch::Device device)
:dtype_(dtype),device_(device){
  // cout<<"binary src init not default"<<endl;
}

torch::Tensor BinarySource::forward(const vector<int64_t>& size){
  return torch::randint(0, 2, size, torch::TensorOptions().device(device_).dtype(dtype_));

}