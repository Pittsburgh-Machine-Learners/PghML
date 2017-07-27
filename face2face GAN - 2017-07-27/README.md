# Face2Face Image Translation with Generative Adversarial Networks

This repo contains the code examples for the [Pittsburgh Machine Learners meetup](https://www.meetup.com/Pittsburgh-Machine-Learners/events/241753177/) on 2017-07-27.

The code is adapted from the pix2pix / face2face TensorFlow implementations with minor modifications. See the face2face README for the procedure to train a model from scratch.

## Environment Setup

To install the python prereqs:

`pip install -r requirements.txt`

If you have an NVIDIA GPU with CUDA installed, you'll want to replace `tensorflow` with `tensorflow-gpu` in the requirements.txt.

The other two major requirements are OpenCV and dlib. OpenCV can be installed from pip, but on MacOS/Linux (and possibly Windows) it lacks GUI support and won't be able to display images. Instructions for compiling these two libs can be found at:


* OpenCV 

  * MacOS - https://medium.com/@satchitananda/setting-up-opencv-for-python-3-on-macos-sierra-with-5-easy-steps-647b64c5e0c9 

  * Linux - http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/ 

* dlib (MacOS & Linux) - http://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

dlib also requires a landmark model to perform facial keypoint detection that can be downloaded [from here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).


## Running a pre-trained model

There is a pretrained face2face model available [from here](https://u7410512.dl.dropboxusercontent.com/u/7410512/face2face-demo/face2face_model_epoch_200.zip). Once that's downloaded, the model can be run with:

```
cd face2face/

python run_webcam.py --landmark-model path/to/shape_predictor_68_face_landmarks.dat --tf-model path/to/frozen_model.pb
```
