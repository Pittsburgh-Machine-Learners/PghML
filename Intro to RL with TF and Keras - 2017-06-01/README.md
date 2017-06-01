# Introduction to Reinforcement Learning with TensorFlow and Keras

This repo contains the code examples for the [Pittsburgh Machine Learners meetup](https://www.meetup.com/Pittsburgh-Machine-Learners/events/240090049/) on 2017-06-01.

The code we'll cover is organized as:

* TensorFlow/

  * **TensorFlow MNIST Tutorial.ipynb** -- Basic example of classifying handwritten digits

  * **TensorFlow Linear Regression.ipynb** -- Linear regression with TF ops

* Cats vs. Dogs/

  * **Cats vs. Dogs with Keras.ipynb** -- Binary classification in Keras with Convolutions and pre-trained nets

* DQN/

  * **dqn.py** -- Train/test a DQN agent to play Breakout in OpenAI Gym. Pre-trained network is included.

  * **ddqn.py** -- Double DQN implementation

  * **TensorBoard Embedding Visualization.ipynb** -- Using the TF Embedding Projector to visualize feature space

  * **Visualizing Conv Net Saliency Maps.ipynb** -- Visualize conv net activations with saliency maps


## Environment Setup

All code examples are written in Python 3. Python 2.7 will also likely work but hasn't been tested.

I recommend using a [virtualenv](https://virtualenv.pypa.io/en/stable/userguide/) virtual environment to isolate the required packages from the rest of the system. This will also help ensure that the correct versions are installed, and makes it easy to get rid of them when no longer needed (just delete the folder).

To install virtualenv:

`pip install virtualenv`

Then:

`virtualenv -p python3 pghml`

Now activate the env (this is different on Windows, see the [user guide](https://virtualenv.pypa.io/en/stable/userguide/)):

`source pghml/bin/activate`

Now we're using the venv's version of python and pip.

Next, install the required packages from the requirements.txt:

`pip install -r requirements.txt`

**NOTE: If you have an NVIDIA GPU you should replace `tensorflow` with `tensorflow-gpu` in requirements.txt. GPU acceleration will significantly speed up some of the code examples.**

If using a virtualenv, there's one more step to get its kernel working with Jupyter notebooks:

```
pip install ipykernel
python -m ipykernel install --user --name pghml --display-name "Python3 PGHML venv"
```

When opening a notebook, change the kernel by going to **Kernel -> Change kernel -> Python3 PGHML venv**
.

Finally, to open Jupyter on port 8888:

`jupyter notebook`

