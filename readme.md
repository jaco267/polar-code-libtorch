#### polar code torch sionna
implement polar code sc decoding in libtorch(c++)    
following the sionna library   

I'm using ubuntu/linux, this haven't been tested on windows or macos      
to run the cpp version you have to first install [libtorch](https://pytorch.org/cppdocs/installing.html), I'm using libtorch version 2.7.0            


```sh  
rm ./build  
mkdir ./build   
cd ./build   
# DCMAKE_PREFIX_PATH should point to the path of libtorch library
cmake -DCMAKE_PREFIX_PATH="~/Path/to/your/libtorch" ..
# ex. I've put libtorch in my downloads folder
#cmake -DCMAKE_PREFIX_PATH="~/Downloads/libtorch" ..
cmake  --build . --config Release
# run polar code with blocklen 64 info bits 32, batchsize 100
./main -n 64 -k 32 -b 100 -i 1

# or if you are lazy you can just run the run.py in that folder
# pip install pyrallis
# python run.py --n 8 --k 4 --bs 3 --iter 1 --libtorch_path "~/Path/to/your/libtorch"
#python run.py --n 8 --k 4 --bs 3 --iter 1 --libtorch_path "~/Downloads/libtorch"
```