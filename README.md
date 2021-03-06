# GeoSeg

## About

This is an API for building neural network for a particular kind of inverse problem arising in geophysics as described in [Hateley, Mylonakis, Roberts, Yang][1]. The frozen Gaussian approximation ([FGA][2]) is a mathematical technique to quickly solve an approximation to the 3D elastic wave equation. The FGA solves for the waveform given an initial epicenter and boundary data. The full inverse problem would be to determine the boundary data and epicenter from the waveform at a specified set of receivers.

The goal of this API is to be able to quickly build different neural network architectures to test increasingly more complicated versions of this inverse problem.

The basic components of models in the package are Blocks and Meta-Architectures. A Block is a combination of network layers  together with an up and down sample method, and a Meta-Architecture is the scaffolding that the blocks will be placed upon. The current supported meta-architectures and blocks are:

- **Meta_Architectures**: 
    - **CNN**: A basic feed forward convolutional network which down samples its input.
    - **EncoderDecorder**: A feed forward network with an encoding branch which down-samples and a decoding branch that up-samples back to original resolution. 
    - [**UNet**][2]: Similar to the auto-encoder but with connections from the encoding branch to the decoding branch. 
- **Blocks**: 
    - **Convolutional**: A single convolutional layer.
    - **[Residual][4]**: A single convolutional layer with a skip connection.
    - **[Dense][3]**: A multi-layer block where each layer receeves input from all previous layers.

The down-sample layer for each block is a strided convolution and their up-sample is a strided convolution-transpose. All blocks have the option of adding a batch-normalization layer and/or a bottleneck as was done in [[3]]. Each of Meta-Architecture can be implemented with any of the Blocks. Details of the implementation can be found in the documentation.

## Running Experiments

To run an experiment you need data seismograph data together with labels in numpy array form. The seismograph data should be of shape (N,3,r) corresponding to N time samples of P or S wave seismic data in the x, y, and z direction recorded by _r_-receivers. 

1. Clone the repo 
~~~
clone https://github.com/KyleMylonakis/GeoSeg
~~~

2. Train the network
~~~
python3 main.py --config path/to/config.json \ 
                          --save-dir path/to/save/model
~~~

At the end of training the model will be in _path/to/save/model/model.h5_ along with the weights from the checkpoints with the highest validation accuracy _path/to/save/model/model_chkpt.h5_.

## Configuring Experiments

Each experiment is defined by an experiment_config.json. There are three main configurations to set.

### Model Config
The model config has a meta_arch and block config. 

~~~
"model":
"meta_arch":{
    "name": "Unet",
    **kwargs   
    },
"block":{
    "name": "dense",
    **kwargs
    }
~~~

The only field required is "name". Any other parameters that define the model will be set to their defaults if not provided. For a description of all the parameters see documentation in meta_arch.MetaModel and blocks.Blocks.

### Train and Eval Config

~~~
"train": {
    "save_every": 100,
    "data": "path/to/train/data.npy",
    "labels": "path/to/train/labels.npy",
    "downsample": 5,
    "optimizer": {
      "algorithm": "nadam",
      "parameters": {
        **kwargs
      }
    },
    "batch_size": 16,
    "epochs": 1,
    "shuffle": true
  },
"eval": {
    "data": "path/to/eval/data.npy",
    "labels": "path/to/eval/labels.npy",
    "batch_size": 16,
    "shuffle": false
  }
~~~


## References
1. [Deep Learning Seismic Interface Detection using the Frozen Gaussian Approximation][1];
James C. Hateley, Jay Roberts, Kyle Mylonakis, Xu Yang. (2018)
2. [UNet: Convolutional Networks for Biomedical Image Segmentation][2]; Olaf Ronneberger, Philipp Fischer, and Thomas Brox. (2015)
3. [Densely Connected Convolutional Networks][3]; Gao Huang, Zhuang Liu. (2018)
4. [Deep Residual Learning for Image Recognition][4]; Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2015)

[1]: https://arxiv.org/abs/1810.06610
[2]: https://arxiv.org/pdf/1505.04597
[3]: https://arxiv.org/pdf/1608.06993
[4]: https://arxiv.org/pdf/1512.03385

