# Neural Style Transfer with TF and Keras

This is the code for the Pittsburgh Machine Learners meetup on February 22nd, 2018. 

The notebook `Neural Style Transfer with TF and Keras.ipynb` implements the algorithm from [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.

To install the prerequisites:

`pip install -r requirements.txt`

If you have an NVIDIA GPU with CUDA/cuDNN installed then uncomment the #tensorflow-gpu line and comment tensorflow. Do the reverse to use the CPU-only version of TF.

To download the VGG16 model ahead of time:

`python3 -c 'from vgg16_avg import VGG16_Avg; VGG16_Avg(include_top=False)`