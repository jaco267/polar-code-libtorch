
#include <iostream>
#include <torch/torch.h>
#include "enc.h"
#include "torch_utils.h"  //"torch_utils/torch_utils.h"
using namespace std;
// Constructor
PolarEncoder::PolarEncoder(//* constructor 
    const torch::Tensor& frozen_pos,
    int n,  
    const torch::Tensor& G  
):G_(G),n_(n),frozen_pos_(frozen_pos){
  // cout<<"enc:frozen pos"<<frozen_pos_.reshape({1,-1})<<"n"<<n_<<endl<<G_.sizes()<<endl;
  k_ = n_ - frozen_pos.sizes()[0];
  info_pos_ =   setdiff1d(torch::arange(n_,torch::kInt64),frozen_pos);
  // cout<<"enc:info pos "<<info_pos_.reshape({1,-1})<<endl;
  assert (k_==info_pos_.sizes()[0] &&"invalid info_pos");
}
torch::Tensor PolarEncoder::forward(
    const torch::Tensor& u) const{
  int bs = u.sizes()[0];
  assert(u.sizes()[1]==k_ && "last dim must be len k");
  auto c=torch::zeros({bs,n_},torch::kInt8);
  // cout<<"test running enc in AWGN..."<<endl;
  // cout<<"input sizes(bs,k)"<<u.sizes()<<endl;
  c.index_put_({"...",info_pos_}, u);
  // cout<<"info_ps"<<info_pos_.reshape({1,-1})<<endl;
  // cout<<"u"<<u<<endl;
  // cout<<"c"<<c<<endl;
  torch::Tensor out_raw = torch::matmul(c,G_);  
  // cout<<"g"<<G_<<endl;  
  // cout<<out_raw<<endl;
  torch::Tensor out = torch::remainder(out_raw,2);
  // cout<<out<<endl;
  return out;
}
