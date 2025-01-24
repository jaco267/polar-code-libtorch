#pragma once
#include <iostream>
#include <torch/torch.h>

using namespace std;

//class need semicolom, function dont need  
class QamConstell{
  public:
    // QamConstell(){QamConstell(2);}; //Qam const default constructor  
    QamConstell(int n_bits_per_symbol); 
    int n_bits_per_sym_; 
    torch::Tensor points_; 
    ~QamConstell(){};  
};
class Mapper{
  public:
    // Mapper(){}; //default constructor  
    Mapper(QamConstell* constell);
    torch::Tensor forward(torch::Tensor& inputs);
  private:  
    QamConstell* constell_;  
    torch::Tensor _binary_base;
};
class SymboLogits2LLRs{
public:
  SymboLogits2LLRs(int n_bits_per_sym);
  torch::Tensor forward(torch::Tensor inputs);
  ~SymboLogits2LLRs(){};
private:
  int n_bits_per_sym_; 
  torch::Tensor c0;  
  torch::Tensor c1;
};

class Demapper{
  public:
    // Demapper(){}; //default const  //don use this use pointer
    Demapper(QamConstell* constell);
    torch::Tensor forward(torch::Tensor& inputs,float no);
  private:  
    QamConstell* constell_;  
    unique_ptr<SymboLogits2LLRs> _logits2llrs;
};