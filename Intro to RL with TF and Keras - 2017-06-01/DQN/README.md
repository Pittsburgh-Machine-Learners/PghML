# Deep Q-Networks with TensorFlow and Keras

This is an implementation of the DQN algorithm from two DeepMind papers:

* [Playing Atari with Deep Reinforcement Learning (2013)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Human-level control through deep reinforcement
learning (2015)](http://files.davidqiu.com/research/nature14236.pdf)

`dqn.py` uses the [OpenAI Gym](https://gym.openai.com/docs) framework to play Breakout. Other games in the Arcade Learning Environment will work by changing the `ENV_NAME` constant at the top and modifying the `Util.preprocess()` method to not crop out the score panel.

To load the pre-trained network:

`python dqn.py -n 5 -e 0.01 -r`

Where the optional params are:

* -n specifies the number of episodes to play
* -e is the epsilon param for probability of taking a random action
* -r is a flag that will record .mp4 videos of each episode

To train a new network:

`python dqn.py -m train`

Also included is an implementation of [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf) in `ddqn.py`. This is identical to dqn.py except for in `Agent.train_network()` where it uses **the current Q-network to select the action** that is then **evaluated by the target network**. This simple modification helps to mitigate overestimation and improve stability. However, it does slow down training because there is an extra inference step to get Q-vals from the current net.

There are some differences from the original DQN paper in this implementation:

* Instead of decaying epsilon to 0.1 over 1 million timesteps, it has a second decay phase down to 0.01 over the following 24 million steps. This was mentioned in a [recent post by OpenAI on DQN baselines](https://blog.openai.com/openai-baselines-dqn/).
* There is no frame-skipping because this is already implemented with random skips by gym.
* The score panel is cropped out and pixels are binarized instead of grayscaled. This helps speed up learning for Breakout but may not be ideal for other games.
* atari-py was recently updated with 4 actions for Breakout (NOOP, FIRE, RIGHT, LEFT) instead of 6 to remove two redundant actions.
