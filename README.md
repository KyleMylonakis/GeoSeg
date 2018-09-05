# Invert FGA

## About

This is an API for building neural network for a particular kind of inverse problem arising in geophysics. The frozen Gaussian approximation ([FGA][1]) is a mathematical technique to quickly solve an approximation to the 3D elastic wave equation. The FGA solves for the waveform given an initial epicenter and boundary data. The full inverse problem would be to determine the boundary data and epicenter from the waveform at a specified set of receivers.

The goal of this API is to be able to quickly build different neural network architectures to test increasingly more complicated versions of this inverse problem. 

The current supported network architectures are UNet, variational autoencoder, fully convolutional, and fully connected. Each of these networks can be implemented with different kinds of basic building blocks for each layer. The current supported blocks are convolutional, residual, and dense. 

## Running the Code

1. Clone the repo 

**THIS STEP WONT WORK SINCE I AM NOT HOSTING ON GITHUB YET**
~~~
clone https://github.com/KyleMylonakis/invert-fga
~~~

2. Process the data for training and evaluation:
~~~
bash prepare_data.bash
~~~

3. Train the network
~~~
python3 runexperiments.py --model-type Unet
~~~

## References
[UNet][1]

[1]: https://arxiv.org/pdf/1505.04597.pdf

