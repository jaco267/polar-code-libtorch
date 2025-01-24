#include <iostream>
#include <torch/torch.h>
#include "enc.h"
#include "awgn_model.h"
#include "awgn.h"
#include "binary_source.h"
#include "mapping.h"
#include "ebno.h"
#include "polar_sc.h"
#include <memory> //for smart pointer 
using namespace std;
void System_Awgn_model::set_awgn(
  int n,int k,PolarEncoder encoder, SC_Dec dec
){
  n_ = n;
  k_ = k;
  encoder_ = encoder;
  dec_ = dec;
  coderate = (float)k_/(float)n_;
  binary_src_ = make_unique<BinarySource>(torch::kInt8, torch::kCPU);
  constell_ = make_unique<QamConstell>(2);
  mapper_ = make_unique<Mapper>(constell_.get());  
  demapper_ = make_unique<Demapper>(constell_.get());
  awgn_channel_ = make_unique<AWGN>();
  return;
}
torch::Tensor System_Awgn_model::forward(
  int batch_size,float ebno_db,torch::Tensor & bits
){
  float no = ebnodb2no(ebno_db,n_bits_per_sym,coderate);
  // cout<<"ebnodb"<<ebno_db<<"no"<<no<<endl;
  vector<int64_t> size = {batch_size, k_};
  bits = binary_src_->forward(size);
  torch::Tensor codewords = encoder_.forward(bits);
  torch::Tensor x = mapper_->forward(codewords);
  // cout<<"mapper out"<<torch::real(x)<<endl;
  // AWGN awgn_test;
  torch::Tensor y = awgn_channel_->forward(x,no);
  torch::Tensor llr = demapper_->forward(y,no);
  // cout<<"llr"<<llr<<endl;
  torch::Tensor bits_hat_ =  dec_.forward(llr);

  return bits_hat_;
}
System_Awgn_model::~System_Awgn_model(){
    // cout<<"del awgn"<<endl;
    //* solution1  will need to delete manually 
    // delete binary_src_;  // new will need to delete by hand   
    // delete constell_;
}