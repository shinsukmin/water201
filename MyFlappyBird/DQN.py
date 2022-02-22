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

class DQN():
    def __init__(self, sess, name, input_size, output_size):
        self.sess = sess
        self.net_name = name
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.0001

        self.createNetwork()

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


    def createNetwork(self):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
        
            W1 = tf.Variable(tf.truncated_normal([self.input_size, 64], stddev = 0.01), name = "w1")
            b1 = tf.Variable(tf.truncated_normal(shape = [64]), name = "b1")
            Layer1 = tf.nn.relu(tf.matmul(self._X, W1) + b1)
            # Layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            W2 = tf.Variable(tf.truncated_normal([64, 32], stddev = 0.01), name = "w2")
            b2 = tf.Variable(tf.truncated_normal(shape = [32]), name = "b2")
            Layer2 = tf.nn.relu(tf.matmul(Layer1, W2) + b2)
            # Layer2 = tf.nn.relu(tf.matmul(Layer1, W2))

            W3 = tf.Variable(tf.truncated_normal([32, 16], stddev = 0.01), name = "w3")
            b3 = tf.Variable(tf.truncated_normal(shape = [16]), name = "b3")
            Layer3 = tf.nn.relu(tf.matmul(Layer2, W3) + b3)
            # Layer3 = tf.nn.relu(tf.matmul(Layer2, W3))

            W4 = tf.Variable(tf.truncated_normal([16, self.output_size], stddev = 0.01), name = "w4")
            self._QPred = tf.matmul(Layer3, W4)

            self._Y = tf.placeholder(tf.float32, shape=[None], name="_Y")
            self._A = tf.placeholder(tf.float32, [None, self.output_size], name="_A")
            
            self._QValue = tf.reduce_sum(tf.multiply(self._QPred, self._A), reduction_indices=1, name="_QValue")

            # loss function
            self._loss = tf.reduce_mean(tf.square(self._Y - self._QValue))
            # learning
            self._train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

    
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.sess.run(self._QPred, feed_dict={self._X: x})
        

    def update(self, s_batch, r_batch, a_batch):
        return self.sess.run([self._loss, self._train], feed_dict={self._X: s_batch, self._Y: r_batch, self._A: a_batch})



        

