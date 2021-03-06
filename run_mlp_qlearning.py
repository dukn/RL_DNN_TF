import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import deque

import game_state as game

class MLPQlearning():
	GAMMA =  0.99 # decay rate of past observation
	OBSERVATION = 10000. # time steps to observe before training
	EXPLORE = 5000. # frames over which to anneal epsilon
	INITIAL_RANDOM_ACTION = 0.05 # starting chance of an action being random
	FINAL_RANDOM_ACTION = 0.05 # final chance of an action being randon
	MEMORY_SIZE = 100000 # number of observations to remember
	MINI_BATCH_SIZE = 100 # size of mini batches
	LEARN_RATE = 1E-5
	INPUT_WIDTH = 60
	INPUT_HEIGHT = 60
	STATE_FRAMES = 4
	ACTION_COUNT =  3

	def __init__ (self, checkpoint_path = "MLP_Q_LEARNING"):

		self._checkpoint_path = checkpoint_path
		self._probability_random_action = self.INITIAL_RANDOM_ACTION
		self._observations = deque()
		self._time = 0
		self._last_action = None
		self._last_state = None
                self._checkpoint_path = checkpoint_path
		self._input_layer, self._output_layer = self._create_network()

		self._action = tf.placeholder("float",[None, self.ACTION_COUNT], name = "actions")
		self._target = tf.placeholder("float",[None],name = "target")

		readout_action = tf.reduce_sum(tf.multiply(self._output_layer, self._action), reduction_indices=1 )

		cost = tf.reduce_mean(tf.square(self._target - readout_action))
		self._train_step = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(cost)

		init = tf.initialize_all_variables()
		self._session = tf.Session()
		self._session.run(init)
		self._saver = tf.train.Saver()

                new_saver = tf.train.import_meta_graph('MLP_Q_LEARNING-100000.meta')
                new_saver.restore(self._session, tf.train.latest_checkpoint('./'))
                all_vars = tf.trainable_variables()
                #if not os.path.exists(self._checkpoint_path):
                #    os.mkdir(self._checkpoint_path)
                #checkpoint = tf.train.get_checkpoint_state(self._checkpoint_path)
                #self._saver.restore(self._session, checkpoint.model_checkpoint_path)
                #checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
                #self._saver.restore(self._session, checkpoint.model_checkpoint_path)

	def _create_network(self):
		input_layer = tf.placeholder ("float",[None,self.INPUT_WIDTH*self.INPUT_HEIGHT*self.STATE_FRAMES],
			name = "input_layer")
		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev = 0.01)
			return tf.Variable(initial)

		def bias_variable(shape):
			initial = tf.constant(0.01, shape = shape)
			return tf.Variable(initial)

		hidden_weights_1 = weight_variable([self.INPUT_WIDTH*self.INPUT_HEIGHT*self.STATE_FRAMES,800])
		hidden_bias_1 = bias_variable([800])

		hidden_weights_2 = weight_variable([800,60])
		hidden_bias_2 = bias_variable([60])

		hidden_weights_3 = weight_variable([60,self.ACTION_COUNT])
		hidden_bias_3 = bias_variable([self.ACTION_COUNT])

		hidden_layer_1 = tf.nn.relu(
			tf.matmul(input_layer,hidden_weights_1) + hidden_bias_1)
		hidden_layer_2 = tf.nn.relu(
			tf.matmul(hidden_layer_1, hidden_weights_2) + hidden_bias_2)
		output_layer = tf.matmul(hidden_layer_2 , hidden_weights_3) + hidden_bias_3

		return input_layer, output_layer

	def get_frame(self,input_screen, reward):
		# convert to grayimage and reshape input
		_, image_input = cv2.threshold(cv2.cvtColor(
			cv2.resize(input_screen,(self.INPUT_WIDTH,self.INPUT_HEIGHT)),
			cv2.COLOR_BGR2GRAY),
			1,255,cv2.THRESH_BINARY)
		image_input = np.reshape(image_input, (self.INPUT_WIDTH * self.INPUT_HEIGHT,))
		# with first frame
		if self._last_state is None :
			self._last_state = np.concatenate( tuple(image_input for _ in range(self.STATE_FRAMES)), axis = 0 )
			random_action = random.randrange(self.ACTION_COUNT)
			self._last_action = np.zeros([self.ACTION_COUNT])
			self._last_action[random_action] = 1

			return random_action

		image_input = np.append(self._last_state [self.INPUT_WIDTH*self.INPUT_HEIGHT:],image_input,axis = 0)
		self._observations.append((self._last_state,self._last_action,reward, image_input))
		if len(self._observations) > self.MEMORY_SIZE:
			self._observations.popleft()

		if len(self._observations) > self.OBSERVATION:
			self._train()
			self._time +=1

		# gradually reduce the probability of a random actionself.
		if self._probability_random_action > self.FINAL_RANDOM_ACTION \
			and len(self._observations) > self.OBSERVATION:
			self._probability_random_action -= \
				(self.INITIAL_RANDOM_ACTION - self. FINAL_RANDOM_ACTION) / self.EXPLORE

		stt = ""
		if len(self._observations) <= self.OBSERVATION:
			stt = "Observation"
		elif self._time <= self.EXPLORE:
			stt = "Explore"
		else:
			stt = "Training"
		#if self._time !=0 and self._time %200 == 0:
	        print "Status: %s, time: %s, random action: %s, reward %s" % (
                        stt,self._time, self._probability_random_action, reward)

		action = self._choose_next_action(image_input)
		self._last_state = image_input
		self._last_action = np.zeros([self.ACTION_COUNT])
		self._last_action[action] =1
		return action

	def _choose_next_action(self, image_input):
		if random.random() <= self._probability_random_action:
			return random.randrange(self.ACTION_COUNT)
		else:
			output = self._session.run(self._output_layer, feed_dict = {self._input_layer : [image_input]})
			return np.argmax(output)

	def _train(self):
		mini_batch = random.sample(self._observations, self.MINI_BATCH_SIZE)
		previous_states = [d[0] for d in mini_batch]
		actions = [d[1] for d in mini_batch]
		rewards = [d[2] for d in mini_batch]
		current_states = [d[3] for d in mini_batch]

		y_expected = []
		y_output = self._session.run(self._output_layer, feed_dict={self._input_layer: current_states})

		for i in range(len(mini_batch)):
			y_expected.append(rewards[i] + self.GAMMA * np.max(y_output[i]))

		self._session.run(self._train_step,feed_dict= {
			self._input_layer: previous_states,
			self._action: actions,
			self._target : y_expected
			})
		#if self._time != 0 and self._time % 20000 == 0:
		#    self._saver.save(self._session, self._checkpoint_path, global_step = self._time)

if __name__ == '__main__':
	player = MLPQlearning()
	game_state = game.GameState()
	do_nothing = np.zeros([player.ACTION_COUNT])
	do_nothing[0] = 1
	image , reward = game_state.gameframe(do_nothing)
	while True:
		action_index = player.get_frame(image,reward)
		action = np.zeros([player.ACTION_COUNT])
		action[action_index] = 1
		image , reward = game_state.gameframe(action)

#
