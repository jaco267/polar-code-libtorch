#pragma once
#include <torch/torch.h>
#include <iostream>
using namespace std;
using namespace torch;
class SC_Dec{
public:
    SC_Dec(){};
    SC_Dec(torch::Tensor frozen_pos, int n);
    Tensor forward(Tensor inputs);
    Tensor f_func(Tensor x, Tensor y);
    Tensor g_func(Tensor x, Tensor y,Tensor u_hat);
    void polar_decode_sc(Tensor cw_ind);
    Tensor decode_batch(Tensor llr_ch);
    ~SC_Dec(){};
private:
    //* init()
    int n_;
    Tensor frozen_pos_; 
    int k_;  
    Tensor info_pos_;  
    float llr_max_;  
    Tensor frozen_ind_;  
    Tensor cw_ind_;  
    int kern_size_;  
    int n_stages_;  
    //* decode_batch  
    Tensor msg_uhat_;  
    Tensor msg_llr_; 
};


