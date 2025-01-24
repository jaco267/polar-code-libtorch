#include "polar_sc.h"
#include <torch/torch.h>
#include "torch_utils.h"
#include <vector>
using namespace std;
using namespace torch;
using namespace torch::indexing;
SC_Dec::SC_Dec(torch::Tensor frozen_pos, int n){
    n_ = n;
    frozen_pos_=frozen_pos; 
    k_ = frozen_pos.sizes()[0];  
    info_pos_ = setdiff1d(torch::arange(n_,torch::kInt64),frozen_pos);
    llr_max_ = 30.;  
    frozen_ind_ = torch::zeros({n_});
    frozen_ind_.index_put_({frozen_pos},1);  
    cw_ind_ = torch::arange({n_});  
    kern_size_ = 2;  
    n_stages_ = (int) log2(n) / log2(kern_size_);
    // cout<<"frzeo_pos"<<frozen_pos<<endl;  
    // cout<<"k"<<k_<<endl;
    // cout<<"info_pos"<<info_pos_<<endl;
    // cout<<"frozne_ind_"<<frozen_ind_<<endl;
    // cout<<"n_stages"<<n_stages_<<endl;  
}
Tensor SC_Dec::f_func(Tensor x, Tensor y){
   auto x_in = torch::clip(x,/*min*/-llr_max_,/*max*/llr_max_);
   auto y_in = torch::clip(y,/*min*/-llr_max_,/*max*/llr_max_);
   //* use max approx  
   auto llr_out = torch::sign(x_in)*torch::sign(y_in)*
     torch::min(torch::abs(x_in),torch::abs(y_in));
   return llr_out;
}
Tensor SC_Dec::g_func(Tensor x, 
   Tensor y,Tensor u_hat){
  auto out = (tensor(1)-tensor(2)*u_hat)*x+y;
  return out;
}
void   SC_Dec::polar_decode_sc(Tensor cw_ind){
  int n = cw_ind.sizes()[0];  
  int stage_ind = log2(n)/log2(kern_size_);
  // cout<<"n"<<n<<"stage_ind"<<stage_ind<<endl; 
  if (n>1){
    Tensor cw_ind_left = cw_ind.index({Slice(0,(int)n/2)});
    Tensor cw_ind_right = cw_ind.index({Slice{(int)n/2,None}});
    //split llr  
    Tensor llr_left = msg_llr_.index({Slice(),stage_ind,cw_ind_left});
    Tensor llr_right = msg_llr_.index({Slice(),stage_ind,cw_ind_right});
    //decode phase i(0~1) + call recur---
    Tensor c_val = f_func(llr_left,llr_right);
    msg_llr_.index_put_({Slice(),stage_ind-1,cw_ind_left},c_val);
    polar_decode_sc(cw_ind_left);
    // cout<<"c_val"<<c_val<<endl;
    Tensor u_hat_left_up= msg_uhat_.index({Slice(),stage_ind-1,cw_ind_left});
    Tensor v_val = g_func(llr_left,llr_right,u_hat_left_up);
    // cout<<"v_val"<<v_val<<endl;
    msg_llr_.index_put_({Slice(),stage_ind-1,cw_ind_right},v_val);
    polar_decode_sc(cw_ind_right);
    //* reencode  
    u_hat_left_up = msg_uhat_.index({Slice(),stage_ind-1,cw_ind_left});
    Tensor u_hat_right_up = msg_uhat_.index({Slice(),stage_ind-1,cw_ind_right});
    Tensor u_hat_left = (u_hat_left_up != u_hat_right_up).to(torch::kFloat); 
    // cout<<"uhat _type"<<u_hat_left<<endl;
    Tensor u_hat0 = torch::concatenate({u_hat_left,u_hat_right_up},/*dim*/-1);
    msg_uhat_.index_put_({Slice(),stage_ind,cw_ind},u_hat0);
  }else{
    // cout<<"bool?"<<frozen_ind_.index({cw_ind})==1<<endl;
    if ((frozen_ind_.index({cw_ind})==1).item<bool>()){
        msg_uhat_.index_put_({Slice(),0,cw_ind},0);
    }else{
       Tensor llr_ch0 = msg_llr_.index({Slice(),0,cw_ind});
       Tensor u_hat = 0.5*(1-torch::sign(llr_ch0));
       msg_uhat_.index_put_({Slice(),0,cw_ind},u_hat);
    }
  }
}
Tensor SC_Dec::decode_batch(Tensor llr_ch){
  // cout<<"yo decode batch"<<endl;
  int bs = llr_ch.sizes()[0];  
  msg_uhat_ = torch::zeros({bs,n_stages_+1,n_});
  msg_llr_ = torch::zeros({bs,n_stages_+1,n_});  
  msg_llr_.index_put_({Slice(),n_stages_,Slice()},llr_ch);
  polar_decode_sc(cw_ind_);
//   cout<<"msg_llr"<<msg_llr_<<endl;s
  return msg_uhat_.index({Slice(),0,Slice()});
}
torch::Tensor SC_Dec::forward(torch::Tensor inputs){
    // cout<<"sc forward is running"<<endl;
    // cout<<"my private n is" <<n_<<endl;
    inputs = inputs.to(torch::kFloat);  
    assert(inputs.sizes()[inputs.dim()-1]==n_ && "last dim must == n");
    // cout<<"sss"<<inputs.sizes()[inputs.dim()-1]<<endl;
    vector<long> input_shape = inputs.sizes().vec();
    auto llr_ch = inputs.reshape({-1,n_});
    llr_ch = -1*llr_ch;  
    auto u_hat_n = decode_batch(llr_ch); 
    // cout<<"final"<<u_hat_n<<endl;
    auto u_hat = u_hat_n.index({"...",info_pos_});  
    // cout<<"u_hat"<<u_hat<<endl;
    vector<long> output_shape = input_shape;
    output_shape[output_shape.size()-1] = k_;
    output_shape[0] = -1;
    // cout<<"output_shape"<<output_shape<<endl;
    auto u_hat_reshape = u_hat.reshape(output_shape);
    // cout<<"u_hat"<<u_hat_reshape<<endl;
    return u_hat_reshape;
}