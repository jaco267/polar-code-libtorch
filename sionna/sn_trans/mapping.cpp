#include "mapping.h"
#include <iostream>
#include <ATen/ATen.h>
#include <vector>
#include "sn_utils.h"
#include "torch_utils.h"
using namespace torch::indexing;
using namespace std;  

torch::Tensor pam_gray(torch::Tensor b){
    if(b.sizes()[0]>1){
       throw std::invalid_argument( "not implement qam bit>1" );
    }
    return 1-2*b[0];//bpsk
}
torch::Tensor qam(int n_bits_per_sym){
    //https://discuss.pytorch.org/t/complex-tensor-in-c-api/117148
    int n_pow = pow(2,(float)n_bits_per_sym);
    torch::Tensor c = torch::zeros({n_pow},
                           torch::dtype(torch::kComplexFloat));
    for (int i=0;i<n_pow;i++){
        torch::Tensor b = binary_repr(i,n_bits_per_sym);//bool type
        torch::Tensor real = b.index({torch::indexing::Slice(0, torch::indexing::None, 2)});
        torch::Tensor imag = b.index({torch::indexing::Slice(1, torch::indexing::None, 2)});
        real = pam_gray(real);
        imag = pam_gray(imag);
        // https://github.com/pytorch/pytorch/issues/73911
        torch::Tensor kk = torch::complex(real.to(at::kFloat),imag.to(at::kFloat));
        c.index_put_({i},kk);
    }
    // c = c+c10::complex<double>(2.0, 3.1);
    // cout<<torch::abs(c)<<"...."<<endl;
    //* normalize  
    int n = n_bits_per_sym / 2; 
    if(n_bits_per_sym > 2){
         throw std::invalid_argument( "not implement err" );
    }
    float qam_var = 2;  
    c /= std::sqrt(qam_var);
    // cout<<"----"<<endl<<torch::real(c)<<endl;
    // cout<<"----"<<endl<<torch::imag(c)<<endl;
    return c;
}

QamConstell::QamConstell(
    int n_bits_per_symbol
):n_bits_per_sym_(n_bits_per_symbol){
    torch::Tensor points = qam(n_bits_per_sym_);
    points_= points;//0.707+0.707j  0.707-0.707j...
    // cout<<"---points_"<<points_<<endl;
}
Mapper::Mapper(QamConstell* constell){ //*mapper constructor
//    cout<<"mapper init"<<endl;
   constell_ = constell;
   //n_bits = 2 case   
   assert(constell->n_bits_per_sym_==2 && "not implement constell n_bit!=2");
//    cout<<"nb per sym"<<constell->n_bits_per_sym_<<endl;
   // cout<<"points"<<torch::real(constell->points_)<<endl;//imag
   _binary_base = torch::tensor({2,1}); ///todo write general  
//    cout<<"bin base"<<_binary_base<<endl;
}
torch::Tensor Mapper::forward(torch::Tensor& inputs){
    // cout<<"mapper forward"<<endl; //* ex (bs,n=8)->(bs,4,2)
    assert (inputs.dim() == 2 && "dim != (bs,n)");
    int nb = constell_->n_bits_per_sym_;
    vector<long> new_shape;  
    new_shape.push_back(-1);  
    c10::ArrayRef<int64_t> ten_size = inputs.sizes();  
    new_shape.push_back(ten_size[1]/nb);
    new_shape.push_back(nb);
    // cout<<"new size"<<new_shape<<endl;
    torch::Tensor inputs_reshaped = inputs.reshape(new_shape);
    // cout<<"input reshape"<<inputs_reshaped<<endl;
    //* convert last dim to an inteber   
    torch::Tensor int_rep = torch::sum(inputs_reshaped*_binary_base,/*dim*/-1);
    // cout<<"int_rep"<<int_rep<<endl;
    torch::Tensor x = constell_->points_.index({int_rep});
    // cout<<"x"<<torch::real(x)<<endl;
    // cout<<"x imag"<<torch::imag(x)<<endl;
    return x;
}

SymboLogits2LLRs::SymboLogits2LLRs(int n_bits_per_sym
 ):n_bits_per_sym_(n_bits_per_sym){
  assert (n_bits_per_sym == 2 && "nbits only support 2");
  int n_points = std::pow(2,n_bits_per_sym);//2^2=4
  // Array composed of binary representations of all symbols indices
  torch::Tensor a = torch::zeros({n_points,n_bits_per_sym});
  for (int i=0; i<n_points;i++){
    /* a = [[0. 0.]     0 
            [0. 1.]     1
            [1. 0.]     2
            [1. 1.]]    3*/
    a.index_put_({Slice(i,None)},//torch::indexing 
        binary_repr(i,n_bits_per_sym));
  }
  // cout<<"hihi a"<<a<<endl;
  c0 = torch::zeros({n_points/2,n_bits_per_sym});
  c1 = torch::zeros({n_points/2,n_bits_per_sym});
  for (int i=n_bits_per_sym-1; i>-1;i--){
    // cout<<".."<<i<<"..ai"<<a.index({Slice(),i})<<endl;
    auto a_equal_0 =  torch::where(a.index({Slice(),i})==0)[0];
    c0.index_put_({Slice(),i}, a_equal_0);
    auto a_equal_1 =  torch::where(a.index({Slice(),i})==1)[0];
    c1.index_put_({Slice(),i}, a_equal_1);
  }

}
torch::Tensor SymboLogits2LLRs::forward(torch::Tensor inputs){
  // cout<<"forwar symlogits2LLRS"<<endl;
  // cout<<"c0"<<c0<<endl;
  // cout<<"c1"<<c1<<endl;
  torch::Tensor exp0 = inputs.index({"...",c0.to(torch::kInt64)});  
  torch::Tensor exp1 = inputs.index({"...",c1.to(torch::kInt64)});
  // cout<<"exp0 sizes"<<exp0.sizes()<<endl;
  // cout<<"exp0"<<exp0<<endl;
  // cout<<"exp1 sizes"<<exp1.sizes()<<endl;
  torch::Tensor llr = torch::logsumexp(exp1,/*dim*/-2)-torch::logsumexp(exp0,-2);                     
  /*c0([[0., 0.],
        [1., 2.]])
    c1([[2., 1.],
        [3., 3.]]*/
  // cout<<"llr"<<llr<<endl;
  return llr;
}
Demapper::Demapper(QamConstell* constell){//*demapper constructor
  constell_ = constell;
  int n_bits_per_sym = constell->n_bits_per_sym_;   
  _logits2llrs = make_unique<SymboLogits2LLRs>(2);
}

torch::Tensor Demapper::forward(torch::Tensor& y,float no){
  // cout<<"y real"<<torch::real(y)<<endl;
  // cout<<"y imag"<<torch::imag(y)<<endl;
  // cout<<"no"<<no<<endl;
  vector<long> points_shape;
  for (int i = 0; i<y.dim(); i++){points_shape.push_back(1);}   
  auto points_shape2 = constell_->points_.sizes().vec();
  points_shape.insert(points_shape.end(),points_shape2.begin(),points_shape2.end());
  torch::Tensor points = constell_->points_.reshape(points_shape);//*4->(1,1,4)
  // Compute squared distances from y to all points
  torch::Tensor squared_dist = torch::pow(torch::abs(y.unsqueeze(/*dim*/-1)-points),2);
//   cout<<"yshape"<<y.unsqueeze(/*dim*/-1).sizes()<<endl;
//   cout<<"square d_dist"<<squared_dist<<endl;
  torch::Tensor no_ten = expand_to_rank(
     torch::tensor(no),/*target_rank*/squared_dist.dim(),/*axis*/-1);
  // cout<<"no shape"<<no_ten.sizes()<<endl;
  torch::Tensor exponents = -squared_dist/no_ten;
  // cout<<"exp"<<exponents<<endl;
  torch::Tensor llr = _logits2llrs->forward(exponents);
  // cout<<"llr"<<llr<<endl;
  vector<long> out_shape = y.sizes().vec();
  long last_shape = out_shape.back();
  out_shape.pop_back();
  // *(long)constell_->n_bits_per_sym_
  out_shape.push_back(last_shape*constell_->n_bits_per_sym_);
  // cout<<"out shape3 "<<out_shape<<endl;
  torch::Tensor llr_reshaped = llr.reshape(out_shape);
  return llr_reshaped;
}

