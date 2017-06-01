# Adapted from:
# * https://github.com/tatsuyaokubo/dqn/
# * https://github.com/gtoubassi/dqn-atari/
# * https://github.com/noahgolmant/DQN/
# * https://github.com/songrotek/DQN-Atari-Tensorflow/
# * https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
# * https://github.com/tambetm/simple_dqn/
# * https://github.com/vuoristo/dqn-agent/blob/master/DQNAgent.py
# * https://github.com/devsisters/DQN-tensorflow
# * https://github.com/openai/baselines

from __future__ import print_function, division
import argparse
import os
import time
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, multiply
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import blosc
import cv2


ENV_NAME = 'Breakout-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor

# Epsilon annealing schedule taken from https://blog.openai.com/openai-baselines-dqn/
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
PHASE1_EXPLORATION_STEPS = 1e6  # Number of steps over which the initial value of epsilon is linearly annealed
PHASE1_FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy for first phase
PHASE2_EXPLORATION_STEPS = 24e6 # Number of steps to anneal in second phase
PHASE2_FINAL_EPSILON = 0.01 # Final value of epsilon

INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 1000000  # Size of replay memory
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
Q_EVAL_INTERVAL = 64 # Controls frequency with which average max Q values are evaluated

LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update

SAVE_INTERVAL = 300000  # The frequency with which the network is saved
SAVE_NETWORK_PATH = 'saved_networks/'
SAVE_SUMMARY_PATH = 'summary/'


class Agent():
    """Implementation of a DQN agent for playing Atari games in OpenAI Gym"""

    def __init__(self,
                 env,
                 mode='train',
                 load_network=False,
                 save_path=SAVE_NETWORK_PATH,
                 epsilon=INITIAL_EPSILON):
        """Initialize a TF session and Q-net/target net graphs.

        Args:
            env: A gym environment for an ALE game.
            mode: 'train' to train a new DQN, else operate in test mode.
            load_network: Load pre-trained network from latest TF checkpoint.
            save_path: Folder path for saving checkpoints.
            epsilon: Prob. of choosing random action. This is annealed in train mode.
        """
        self.env = env

        self.mode = mode

        self.save_path = save_path
        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(save_path)

        self.epsilon = epsilon

        # Step size for annealing epsilon in phase 1
        self.phase1_epsilon_step = (INITIAL_EPSILON - PHASE1_FINAL_EPSILON) / PHASE1_EXPLORATION_STEPS
        # Step size in phase 2
        self.phase2_epsilon_step = (PHASE1_FINAL_EPSILON - PHASE2_FINAL_EPSILON) / PHASE2_EXPLORATION_STEPS

        self.num_actions = self.env.action_space.n

        print("Environment {} has {} actions".format(env, self.num_actions))

        self.t = 0

        # Parameters used for TensorBoard summary
        self.total_reward = 0
        self.rewards_clipped = 0
        self.total_q_max = 0
        self.total_q_max_count = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Setup replay memory
        self.replay_memory = deque(maxlen=NUM_REPLAY_MEMORY)

        # Create q network
        self.s, self.q_values, self.q_network = self.build_network()
        q_network_weights = self.q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)

        # Create TF session and limit GPU memory usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

        # Set Keras to use the same sess
        K.set_session(self.sess)

        # Setup model saver
        self.saver = tf.train.Saver(q_network_weights, max_to_keep=None)

        if load_network:
            # Load pre-trained network from latest TF checkpoint
            self.load_network()

        if self.mode == 'train':
            # Setup summary writing for TensorBoard
            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

            # Initialize target network from q_net weights
            self.sess.run(self.update_target_network)

    def build_network(self):
        """Construct the Q-network graph with Keras layers

        Returns:
            s: placeholder for feeding state batches.
            q_values: Output tensor for inferring Q-vals.
            model: Keras model.
        """
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])

        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        """Construct loss function and RMSProp optimization ops.

        Args:
            q_network_weights: List of weights to be trained.

        Returns:
            a: placeholder for feeding chosen actions.
            y: placeholder for target values.
            loss: Tensor output for Huber loss.
            grads_update: RMSProp update op to minimize the loss.
        """
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert chosen action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # Multiply to select q-vals for actions taken
        q_acted = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), axis=1)

        # Temporal-difference error between target and predicted q-val
        td_error = y - q_acted

        def huber_loss(x, delta=1.0):
            """Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region

            See https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/ for explanation of why this is used instead of MSE
           
            Reference: https://en.wikipedia.org/wiki/Huber_loss
           
            Borrowed from: https://github.com/openai/baselines/blob/958810ed1e78624c300e327a0c79212f2453cfb7/baselines/common/tf_util.py
            """
            return tf.where(
                tf.abs(x) < delta,
                tf.square(x) * 0.5,
                delta * (tf.abs(x) - 0.5 * delta)
            )

        loss = tf.reduce_mean(huber_loss(td_error))

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def load_network(self):
        """Load network weights from latest checkpoint"""
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)

    def get_action(self, state):
        """Choose an eps-greedy action for a state.

        Args:
            state: State tensor with shape (84,84,4)

        Returns:
            action: Index of chosen action.
        """
        if self.mode == 'train':
            if random.random() <= self.epsilon or self.t < INITIAL_REPLAY_SIZE:
                # In the initial 'random' phase choose a random action
                action = self.env.action_space.sample()
            else:
                # In explore/exploit phases choose the action with max q-val
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))
        else:
            # Test mode
            if random.random() <= self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [state]}))
            self.t += 1

        return action

    def update(self, state, action, reward, next_state, terminal):
        """Store the s,a,r,s',t tuple in replay memory and periodically train/update stats.

        Args:
            state: Tensor of shape (84,84,4) at time t.
            action: Index of chosen action.
            reward: Scalar (unclipped) reward value.
            next_state: State tensor at time t+1.
            terminal: Bool indicating end of episode.
        """
        self.duration += 1    # Update counter for episode duration

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward_clipped = np.clip(reward, -1, 1)
        self.rewards_clipped += reward_clipped

        # Store transition in replay memory
        self.replay_memory.append((Util.compress(state), action, reward_clipped, Util.compress(next_state), terminal))

        if self.t >= INITIAL_REPLAY_SIZE:  # Wait until we've collected random experience
            # Periodically train network with minibatches from replay memory
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network periodically
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network periodically
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

            # Anneal epsilon linearly over time during scheduled exploration phases
            if self.epsilon > PHASE1_FINAL_EPSILON:
                self.epsilon -= self.phase1_epsilon_step
            elif self.epsilon > PHASE2_FINAL_EPSILON:
                self.epsilon -= self.phase2_epsilon_step

        # Track reward for this episode
        self.total_reward += reward

        # To speed things up, only periodically evaluate Q values
        if self.t % Q_EVAL_INTERVAL == 0:
            self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [state]}))
            self.total_q_max_count += 1       

        if terminal:    # We're done with the episode, write stats and clean up
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.rewards_clipped, self.total_q_max / float(self.total_q_max_count),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), self.episode + 1, self.epsilon]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.t)

            # Debug to console
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + PHASE1_EXPLORATION_STEPS:
                mode = 'explore(1)'
            elif self.t < INITIAL_REPLAY_SIZE + PHASE1_EXPLORATION_STEPS + PHASE2_EXPLORATION_STEPS:
                mode = 'explore(2)'
            else:
                mode = 'exploit'

            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.total_q_max_count),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            # Reset counters
            self.total_reward = 0
            self.rewards_clipped = 0
            self.total_q_max = 0
            self.total_q_max_count = 0
            self.total_loss = 0
            self.duration = 0
            # Update episode count
            self.episode += 1

        # Update overall timestep count
        self.t += 1

    def train_network(self):
        """Sample a minibatch from replay memory and run a training step"""
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(Util.decompress(data[0]))
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(Util.decompress(data[3]))
            terminal_batch.append(data[4])

        # Convert from bools to ints
        terminal_batch = np.array(terminal_batch) + 0

        # DOUBLE Q - Use the *Q-net* to choose the best action for the *target* net to eval
        q_select_action = self.q_values.eval(feed_dict={self.s: np.float32(np.array(state_batch))})
        next_action_batch = np.argmax(q_select_action, axis=1)

        # Get target q-vals from target network
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})

        # Bellman target. Target Q's indexed by the action selected by Q-net.
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * \
                    target_q_values_batch[np.arange(len(target_q_values_batch)), next_action_batch]

        # Run the update step and retrieve scalar loss
        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        """Setup placeholders and ops for writing TensorBoard summaries"""
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_reward_clipped = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Reward Clipped/Episode', episode_reward_clipped)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        episode_num = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Episode', episode_num)
        epsilon = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Epsilon', epsilon)
        summary_vars = [episode_total_reward, episode_reward_clipped, episode_avg_max_q, episode_duration, episode_avg_loss, episode_num, epsilon]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


class Util(object):
    """Helpers for the Agent"""

    @staticmethod
    def compress(image):
        return blosc.compress(np.reshape(image, 84 * 84 * STATE_LENGTH).tobytes(), typesize=1)

    @staticmethod
    def decompress(image):
        return np.reshape(np.fromstring(blosc.decompress(image), dtype=np.uint8), (84, 84, STATE_LENGTH))

    @staticmethod
    def preprocess(observation):
        """Preprocess RGB observation to crop and binarize."""
        # Convert to grayscale
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        # Crop out the score panel
        observation = observation[26:110,:]
        # Convert non-zero pixels to 1
        ret, observation = cv2.threshold(observation,1,1,cv2.THRESH_BINARY)
        return np.reshape(observation, (84,84,1))


def main():
    parser = argparse.ArgumentParser(description='DQN for playing Breakout')
    parser.add_argument('-m','--mode', help='train / test', default='test')
    parser.add_argument('-r','--record', dest='record', action='store_true', help='Record episodes to .mp4')
    parser.add_argument('-n','--num-episodes', help='# of episodes to play in test mode', default=5, type=int)
    parser.add_argument('-e','--epsilon', help='Epsilon to use for test mode', default=0.01, type=float)
    args = parser.parse_args()

    # Instantiate gym environment
    env = gym.make(ENV_NAME)   

    if args.mode == 'train':  # Train mode
        print("Training new network")

        # Create our agent
        agent = Agent(env, mode='train', load_network=False)
   
        while True:    # Train forever
            terminal = False
            observation = env.reset()

            # Do nothing for random # of steps
            for _ in range(random.randint(1, NO_OP_STEPS)):
                observation, _, _, _ = env.step(0)  # No-op

            # Convert observation to binary and stack
            state = Util.preprocess(observation)
            state = np.stack([state]*STATE_LENGTH, axis=2).reshape((84,84,4))

            while not terminal:
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()  # Uncomment to see it train, will be much slower

                next_state = Util.preprocess(observation)

                # Pop the oldest frame and append the processed new frame
                next_state = np.append(state, next_state, axis=2)[:,:,1:]

                # Update stats, store transition, and (periodically) train network
                agent.update(state, action, reward, next_state, terminal)

                state = next_state
   
    elif args.mode == 'test':  # Test mode with pretrained network
        if args.record:
            env = gym.wrappers.Monitor(env, ENV_NAME+'-videos', video_callable=lambda episode_id: True, force=True)

        # Create our agent
        print("Loading pretrained network")
        agent = Agent(env,
                      mode='test',
                      load_network=True,
                      epsilon=args.epsilon)

        for _ in range(args.num_episodes):
            terminal = False
            observation = env.reset()

            # Do nothing for random # of steps
            for _ in range(random.randint(1, NO_OP_STEPS)):
                observation, _, _, _ = env.step(0)  # No-op

            total_reward = 0
           
            state = Util.preprocess(observation)
            state = np.stack([state]*STATE_LENGTH, axis=2).reshape((84,84,4))

            while not terminal:
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
               
                env.render()
               
                next_state = Util.preprocess(observation)

                # Pop the oldest frame and append the processed new frame
                next_state = np.append(state, next_state, axis=2)[:,:,1:]

                state = next_state

                total_reward += reward

            print("Total reward for episode:", total_reward)


if __name__ == '__main__':
    main()
