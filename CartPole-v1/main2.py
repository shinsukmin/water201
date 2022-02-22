#  https://jonghyunho.github.io/reinforcement/learning/cartpole-reinforcement-learning.html

import gym
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(64, activation="relu")
        self.fc1_2 = Dense(32, activation="relu")
        self.fc2 = Dense(16, activation="relu")
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc1_2(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        # self.epsilon = 1.0
        self.epsilon = 0.01
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 200
        # 리플레이 메모리는 최대 크기 2000으로 설정하였다.
        self.memory = deque(maxlen=2000)
        #  model과 target_model 두개의 인공신경망을 생성한다. Q함수를 학습하기 위해 model의 파라미터가 학습도중 갱신되는데,
        #  이 파라미터의 변경으로 인하여 정답으로 간주되는 다음 상태의 Q함수도 함께 변경이 된다. 이를 막기 위해 다음 상태의
        #  Q함수를 위한 별도의 target_model을 분리하여 사용한다.
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr=self.learning_rate)

        # self.load_weights()
        # self.update_target_model()

    def load_weights(self):
        self.model.load_weights('./save_model/model')
        self.target_model.load_weights('./save_model/model')


    def update_target_model(self):
        # target_model의 가중치를 model의 가중치로 업데이트 하는 함수이다. 일정 주기로 타켓 흔들림을 해결하기 위해
        weight = self.model.get_weights()
        self.target_model.set_weights(weight)

    # 리플레이 메모리에 현재 상태 S, 액션 A, 보상 R, 다음 상태 S', 완료여부 done을 저장한다.
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #  epsilon을 이용하여 탐험(Exploration)와 활용(Exploitation)의 비율을 조정한다.
    #  학습된 정보만을 이용하여 action을 선택하게 되면 새로운 환경에 대해 경험해 볼 수 없기 때문에 랜덤한 수를 골라 e보다 작으면 랜덤,
    #  그렇지 않으면 학습된 모델을 사용하는 E0greedy 정책을 사용한다.
    def choose_action(self, state):
        if (np.random.rand() <= self.epsilon):
            return random.randrange(self.action_size)
        else:
            qs = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
            return np.argmax(qs.numpy())

    def sample_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        return states, actions, rewards, next_states, dones

    ## computing TD target: y_k = r_k + gamma* max Q(s_k+1, a)
    def td_target(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.discount_factor * max_q[i]
        return y_k

    ## single gradient update on a single batch data
    def dqn_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_size)
            q = self.model(states)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    
    # 샘플 간 correlation을 줄이기 위해 리플레이 메모리에 저장된 데이터를 랜덤으로 섞어 훈련에 사용할 미니 배치 데이터를 생성한다.
    # 벨만  최적 방적식을 이용하여 계산된 정답에 해당하는 targets와 예상 값 predicts 의 차이를 줄여 나가는 경사 하강법으로 학습을 진행한다.
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        states, actions, rewards, next_states, dones = self.sample_batch()

        # predict target Q-values
        target_qs = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))

        # compute TD targets
        y_i = self.td_target(rewards, target_qs.numpy(), dones)

        # train critic using sampled batch
        self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                   actions,
                                   tf.convert_to_tensor(y_i, dtype=tf.float32))
        

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0
    # episode 가 시작될 때마다 환경을 초기화한다.
    num_episode = 300
    for e in range(num_episode):
        done = False
        score = 0

        state = env.reset()

        agent.model(tf.convert_to_tensor([state], dtype=tf.float32))
        agent.target_model(tf.convert_to_tensor([state], dtype=tf.float32))

        # 현재 상태에서 action 을 하나 선택하여 한 스텝 진행한다.
        # 그 결과로 받은 보상을 현재 상태와 선택한 행동과 함께 리플레이 메모리에 저장한다.
        # 리플레이 메모리가 일정 크기 이상으로 저장되면 매 스텝마다 학습할 수 있도록 한다.
        while not done:
            env.render()

            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            score += reward
            if not done or score == 500:
                reward = 0.1
            else:
                reward = -1
                
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()

                if score_avg == 0:
                    score_avg = score
                else:
                    score_avg = 0.9 * score_avg + 0.1 * score
                
                print('episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f}'.format(e, score_avg, len(agent.memory), agent.epsilon))

                scores.append(score_avg)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel('episode')
                plt.ylabel('average score')
                plt.savefig('cartpole_graph.png')

                if score_avg > 400:
                    agent.model.save_weights('./save_model/model', save_format='tf')
                    sys.exit()



