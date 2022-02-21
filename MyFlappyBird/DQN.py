# DeepLearningFlappyBird 코드를 내임맛에 맞게 수정해보자.
# 
# 
# 
# 
import tensorflow as tf
import numpy as np

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 50000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.00001 # final value of epsilon
INITIAL_EPSILON = 0.00001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
USE_CHECK_POINT = True

print(np.__version__)
class DQN():
    def __init__(self):
        print('Do nothing')

    # truncate_normal: 절단정규분포로부터의 난수값을 반환합니다.
    # stddev: 0-D 텐서 또는 파이썬 값. 절단정규분포의 표준 편차.
    # https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/constant_op.html
    # shape 형태의 행렬을 정규분포의 난수로 채워넣음.
    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def createNetwork(self):
        W_conv1 = DQN.weight_variable([8, 8, 4, 32])

        b_conv1 = DQN.bias_variable([32])

        W_conv2 = DQN.weight_variable([4, 4, 32, 64])
        b_conv2 = DQN.bias_variable([64])

        W_conv3 = DQN.weight_variable([3, 3, 64, 64])
        b_conv3 = DQN.bias_variable([64])

        W_fc1 = DQN.weight_variable([1600, 512])
        b_fc1 = DQN.bias_variable([512])

        W_fc2 = DQN.weight_variable([512, ACTIONS])
        b_fc2 = DQN.bias_variable([ACTIONS])

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(DQN.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = DQN.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(DQN.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(DQN.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        self.s = s
        self.readout = tf.matmul(h_fc1, W_fc2) + b_fc2
