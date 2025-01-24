#pragma once  
#include "enc.h"          //polar/enc
#include "binary_source.h"//sionna/trans
#include "mapping.h"      //sionna/trans
#include "awgn.h"  //sionna/trans/models
#include "polar_sc.h"   

#include <torch/torch.h>
#include <memory>   //for smart pointer  
class System_Awgn_model{
public:
  System_Awgn_model(){}; //default const
  void set_awgn(int n, int k, PolarEncoder enc, SC_Dec dec);
  torch::Tensor forward(int batch_size,
    float ebno_db,torch::Tensor & bits);
  ~System_Awgn_model();
private:
    /* data */
  int n_bits_per_sym = 2; 
  int n_;  int k_;  
  float coderate;
  //* ----solution 1:new -----------
  // QamConstell* constell_;   
  //* ----solution 2: unique----------  
  unique_ptr<QamConstell> constell_;//* sol 2 unique pointer  
    // https://www.youtube.com/watch?v=UOB7-B2MfwA
    //* I prefer unique_ptr in this case , since we only have one constell_  (dont need copy)
    //* so we dont need to use shared_ptr ref count overhead  
  //* ---solution 3: shared------
  // shared_ptr<QamConstell> constell_;
  // Mapper* mapper_;   Demapper* demapper_; 
  unique_ptr<Mapper> mapper_;
  unique_ptr<Demapper> demapper_;
  // BinarySource* binary_src_; 
  unique_ptr<BinarySource> binary_src_;
  unique_ptr<AWGN> awgn_channel_;
  // BinarySource binary_sss;//* this will call default constructor
  PolarEncoder encoder_;
  SC_Dec dec_;
};


