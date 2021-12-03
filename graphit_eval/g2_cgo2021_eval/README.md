# G2_artifact_eval

## Introduction
This repository is the guide for evaluating our CGO2021 paper, "Techniques for Compiling Graphs Algorithms for GPUs". This guide has steps for cloning, compiling and executing the implementation of the compiler framework G2 which is built on top of the [GraphIt DSL compiler](https://graphit-lang.org/). 
This guide has two parts -
  - Part 1: Reproducing Figure 3. in the paper to demonstrate that the compiler can generate very different optimized code for different schedules
  - Part 2: Reproducing the G2 columns for all the applications and graphs from Table 7 to demonstrate the performance of the code generated from the G2 compiler

Since Table 7 shows the performance numbers when run on the NVIDIA-Tesla V-100 GPU, the exact execution times you will get in Part 2 will depend on the actual GPU you use. If you do not have access to the same GPU, we have provided access to our system with this GPU in our artifact evaluation submission. 
If you use any other GPU the schedules might have to be tuned to get the best performance for the GPU. 

## Requirements
We expect you to run the artifact evaluation on a Linux system with at least 40GBs of space. Following are the software requirements for each of the parts

### Part 1
Since part 1 only demonstrates the different code generated for different schedules, this part does *NOT* require an NVIDIA GPU or CUDA to be installed. The only software requirements are - 

 - cmake (>= 3.5.1)
 - CXX compiler (like g++ >= 5.4.0)
 - python3 
 - make
 - bash
 - git

### Part 2
Part 2 demonstrates the performance of these applications on the actual GPU. Ideally we require an NVIDIA Tesla V-100 for best results, but other NVIDIA GPUs would also work (the actual performance numbers would be different in that case). Following are the requirements besides all the requirements from Part 1 - 

 - NVIDIA GPU (Pascal generation or better, preferred NVIDIA Tesla V-100 32 GB, access to our machine provided in the artifact evaluation submission). 
 - CUDA SDK (>= 9.0)


## How to run 

### Cloning
We will start by cloning this repository on the evaluation system using the following command - 

    git clone --recursive https://github.com/AjayBrahmakshatriya/G2_artifact_eval.git

If you have already cloned this repository without the `--recursive` command you can get the submodules by running the following commands. Otherwise you can directly proceed to Building G2.

    git submodule init
    git submodule update
   
### Building G2
Start by navigating to the `G2_artifact_eval` directory. We will first build the G2 compiler by running the following commands from the repo's top level directory - 

    cd graphit
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    
If no errors are reported, the G2 compiler is built correctly and you can navigate back to the repository's top level directory and proceed to "Running Part 1"

### Running Part 1
With the G2 compiler built, you can run the Part 1 to generate the code as shown in Figure 3. You can start by returning to the top level directory of the repository with the command `cd ../../` and then running the command - 

    python3 gen_fig3.py

When running this command, the program will prompt for a few options like paths to where the G2 compiler is built and the output directory path. If you have followed the above steps, you can simply press enter and choose the default options shown in `[]`. 

This command should take about 5 mins to run and if it doesn't report any errors, the appropriate files have been generated. Notice the above commands also prints all the commands that were executed to generate the output files. 

The source files for the three schedules in Figure 3 are in the `fig3_inputs/` directory - `fig3_inputs/fig3_a.gt`, `fig3_inputs/fig3_b.gt` and `fig3_inputs/fig3_c.gt`. You can open and read them in your favorite text editor. All the three programs have the same algorithm input but different schedules a the bottom under the `schedule:` section. You can match this schedule with the one in the paper (barring some syntactic changes in the paper for brevity). 

If you choose the default options while running the above programs, the outputs should be generated in the `fig3_outputs/` directory - `fig3_outputs/fig3_a.gt.cu`, `fig3_outputs/fig3_b.gt.cu` and `fig3_outputs/fig3_c.gt.cu`. Again, you can open and read them in your favorite text editor or simple `cat` them. 

You can match the body of the `main` and the user defined function `updateEdges`. Again, we have changed the syntax a little in the paper for brevity. 

### Obtaining the datasets
The datasets are only required for Part 2. If you are not planning to run Part 2, you do not need to obtain the datasets.

We have created two datasets for your convenience - *small* and *all*. The small dataset contains just two graphs (one with bounded degree distribution and one with power law degree distribution). Obtaining and running the small dataset should take less than 15 mins and quickly tests all the variants for all algorithms. The all dataset contains all the 9 graphs from the paper and would take much longer to run (upwards of 1.5 hours on our system). 

There are two ways of obtaining the datasets. If you are running this artifact evaluation on the system we have provided access to, you can quickly fetch all the data set files by running the following commands in the top level directory - 

    cd dataset
    make local
    
If everything succeeds, the dataset should be soft-linked into this directory and you can verify that by running the `ls` command. You can now navigate to the top level directory using the command `cd ../` and proceed to the next step ("Running Part 2"). If the command reports the error 

> You are not running this command on the right host. Please use `make dataset` instead

it means that you are not running the artifact evaluation on our system and you should use the other method for downloading the datasets

If you are running the artifact evaluation on your own system, the script will have to download a tar ball and extact the files. We have a separate command for *small* and *all* datasets. So if you are planning to run the evaluation only for the 2 graphs, please download only the small dataset to save time.

For downloading the *all* dataset run the following commands from the top-level directory -  

    cd datasets
    make dataset

To download just the *small* dataset run the following commands from the top-level directory - 

    cd datasets
    make small

    
This step will take some time, because it downloads and uncompresses all the datasets. After the command succeeds, you can verify that the files are downloaded by running the `ls` command. The small dataset is part of the all dataset and if you accidently downloaded the all dataset, you can still run the small part of the experiment. 

Navigate to the top level directory in any case using the command - `cd ../`

### Running Part 2
This part evaluates the generated code for all the applications and inputs to reproduce Table 7 in the paper. A reminder that if you are running the experiments on a system with any other GPU than the NVIDIA Tesla V-100 (32 GB), the results might be different. The system we have provided with the artifact evaluation has the correct GPU. 

Before we actually run the evaluation, we will list all the GPUs in the system and find one that is completely free. We need a free GPU because the performance might be hampered if other processes are running on the same GPU. 

Start by running the command - 

    nvidia-smi
    
This will list all the GPUs attached to the system numbered from 0. At the bottom of the table, there is a Processes section which shows what processes are running on which GPU. Find a GPU which doesn't have any processes running on it and note down its ID. Suppose for the purpose of this evaluation, the 4th GPU (ID: 3) is free and we want to use that. 

We do not recommend running the evaluation on a GPU that is being used by other processes since it might affect the evaluation results (and correctness) a lot. 

Before running the actual command for running all the experiments, make sure you have successfully built G2 and fetched the datasets. If you are planning to run the *all* dataset make sure you have downloaded the entire dataset. 

To run only the small data set navigate to the top level directory of the repository and run the command - 

    python3 gen_table7.py small

To run the all data set navigate to the top level directory of the repository and run the command - 

    python3 gen_table7.py
    
Again, like Part 1 the program will prompt for various options like path to the CUDA compiler, CXX compiler, path to G2 build directory and the GPU to use. Following is the description of each of the options - 

- Output directory to use: This is the directory where the output of this section will be generated. Please select the default option by pressing enter (notice that the outputs from previous runs will be wiped. So if are planning to run multiple times and want to preserve old results, copy the results somewhere else). 
- GraphIt build directory: This is the path to the `build/` directory where G2 is compiled. If you have followed the exact steps above, just choose the default by pressing enter. 
- Dataset path: This is the directory where the datasets are fetched. If you have followed the exact steps mentioned above, just select the default by pressing enter. 
- NVCC path: This is the path to the `nvcc` compiler from the CUDA SDK. Typically this binary is located at `/usr/local/cuda/bin/nvcc`. If you have installed it elsewhere, please provide the path here. If you have the binary in your `$PATH` variable (you can verify this by running `nvcc --version`), you can simply type `nvcc` and press enter. If you are using the system that we have provided, just press the enter key. 
- CXX_COMPILER path: This is the path to the CXX compiler that you want to use. The default option is `/usr/bin/g++`. If you are using a different compiler, please provide the path here. If you are using the system that we have provided, just press the enter key. 
- GPU ID to use: This is the GPU ID that you want to use to run the experiments on. We have obtained the ID of a GPU that is free in the above step. Enter that here. If the 4th GPU (ID: 3) is free, type `3` and press enter. The default option is `0`, but `0` might not be free. 

Once you enter all the options, the experiments will run one after the other. The program will print which application it is currently running and how many graphs it is done evaluating on. Sit back because running all the applications can take a while (~20 mins for small dataset and >1.5 hrs for all dataset)

If the program completes execution without any errors, all the experiments are done and you can view the final results in the output directory. If you chose the default option, the output file should be under `table7_outputs/table7.txt`. The program should also print the table on successful completion. 


## Evaluating related works
Table 7 in the paper also compares against other related works to compare the speedups we obtain with our compiler. Unfortunately the source code of some of the related works is not directly usable (we found some bugs in the systems and had to fix them ourselves). But the related work Gunrock has a system that is easy to build and evaluate - 

The source code for Gunrock is available in the repository - 

    https://github.com/gunrock/gunrock

Building gunrock is very easy and you can simply follow the instructions under "Quick Start Guide". Gunrock also provides access to docker containers that should have all the dependences packaged. You can use those if it is more convenient. 

If you successfully build gunrock, you can run the experiments with the same dataset that you obtained for G2 evaluation (mtx file format). 

For example to run the pagerank application with Gunrock, the binary is located in the `build/bin/` directory. You can run it using the command - 

    ./pr market /path/to/dataset/roadNet-CA.mtx --num-runs=10 --quick --device=3 --advance-mode=TWC

Here `--device=3` is the GPU to run the experiments on. Change the ID to the proper GPU you want to run on. For other applications like BFS and BC some extra parameters like `do-bfs-a` and `do-bfs-b` are required. The details of all applications, their command line arguments and reference results can be found at - 

    https://gunrock.github.io/docs/#/analysis/engines_topc/engines_topc_table 


