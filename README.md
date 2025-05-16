# Source code for the Reproducibility Challenge for the paper "Learning the greatest common divisor: explaining transformer predictions"

This directory contains the source code for reproducing the results of the paper [Learning the greatest common divisor: explaining transformer predictions](https://arxiv.org/abs/2308.15594) (ICLR 2024).

## Environment for Local GCD
* Requirements: Python 3.12.9, Pytorch 2.7.0, Numpy 2.0.0.
* OS: Tested on Windows
* A NVIDIA/CUDA GPU is required if you intend to train models. This code will not run on Apple MPS.

## Environment for Iridis GCD
* Requirements: Python 3.13.2, Pytorch 2.6pip .0, Numpy 2.2.5.
* OS: Iridis HPC 
* Ran on V100 GPU in GPU partition.

## Running Local GCD
To run the program: open filebase in GCD Local as a Jupyter notebook and run all. 

## Important Parameters for Local GCD

`validate_step`: The amount of epochs passes before the a uniformly sampled outcome dataset is generated to validate the model, default is 1.

`layers`: The number of layers in the transformer model, default is 2.

`heads`: The number of heads in the attention mechanism, default is 4.

`hidden_dimension`: The embedding dimension, default is 256.

`length`: Maximum length of the sequence, default is 512.

`lr`: The learning rate of the optimiser, default is 10e-5.

`batch`: The batch size used for training, default is 128.

`max_int`: The maximum number possibily sample for inputs, default is 1000000.

`sample_size`: The number of samples generated per epoch, default is 10000.

`dropout`: The dropout rate, default is 0.

`max_epoch`: The maximum epoch, default is 100.

`base`: The numerical base used to encode the inputs and outputs, default is 30.

`mix`: Training the model using a mix of uniformly sampled inputs and unformly sampled outputs, defaut is True.

## Running Iridis GCD

* Create Python enviroment with correct requirements.
* If you want to change important paramenters such as bases and distributions that models are trained on edit train.py
* Create script to push job to Iridis GPU to run train.py. Can use template provided.
* Training loop trains a model on each base with each different distribution for the desired number of epochs. Outputs from the training loop are: json file containing final training metrics, log of training metrics for entire training loop, and pickle of the trained model.     

## Important Parameters for Iridis GCD

`validate_step`: The amount of epochs passes before the a uniformly sampled outcome dataset is generated to validate the model, default is 1.

`layers`: The number of layers in the transformer model, default is 2.

`heads`: The number of heads in the attention mechanism, default is 4.

`hidden_dimension`: The embedding dimension, default is 256.

`length`: Maximum length of the sequence, default is 512.

`lr`: The learning rate of the optimiser, default is 10e-5.

`batch`: The batch size used for training, default is 128.

`max_int`: The maximum number possibily sample for inputs, default is 1000000.

`sample_size`: The number of samples generated per epoch, default is 10000.

`epochs`: The number of epochs that each model is trained for.

`bases`: List that contains the bases that model is to be trained on default is 2, 3, 4, 5, 6, 7, 10, 11, 12, 16, 20, 36.

`distributions`: List that contains distributions that models are trained on default is "uniform", "normal", "loguniform", "poisson",
                 "geometric", "exponential".


## References - citations

Learning the greatest common divisor: explaining transformer predictions

`@misc{charton2023GCD,
  url = {https://arxiv.org/abs/2308.15594},
  author = {Charton, François},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learning the greatest common divisor: explaining transformer predictions},
  publisher = {arXiv},
  year = {2023}
}`
