# DeepLearningFlappyBird 코드를 내임맛에 맞게 수정해보자.
# 
# 
# 
# 
import tensorflow as tf
import sys
import gym
from collections import deque
import random

from DQN import DQN
sys.path.append("game/")
import numpy as np

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

discount_factor = 0.99
epsilon = 0.1
epsilon_decay = 0.999
epsilon_min = 0.001
batch_size = 64
train_start = 200
memory = deque(maxlen=4000)

sess = tf.InteractiveSession()
mainDQN = DQN(sess, "main", state_size, action_size)

sess.run(tf.initialize_all_variables())

def choose_action(state):
    if (np.random.rand() <= epsilon):
        return random.randrange(action_size)
    else:
        return np.argmax(mainDQN.predict(state))

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def trainModel():
    global epsilon
    if (epsilon > epsilon_min):
        epsilon = epsilon * epsilon_decay

    s_batch = []
    r_stack = []
    a_batch = []
    
    
    mini_batch = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in mini_batch:
        if done:
            r_stack.append(reward)
        else:
            r_stack.append(reward + discount_factor * np.max(mainDQN.predict(next_state)))

        s_batch.append(state)
        a_onehot = [0, 0]
        a_onehot[action] = 1
        a_batch.append(a_onehot)

    return mainDQN.update(s_batch, r_stack, a_batch)


def playGame():

    scores, episodes = [], []
    score_avg = 0
    # episode 가 시작될 때마다 환경을 초기화한다.
    num_episode = 3000
    for e in range(num_episode):
        done = False
        score = 0

        state = env.reset()

        # 현재 상태에서 action 을 하나 선택하여 한 스텝 진행한다.
        # 그 결과로 받은 보상을 현재 상태와 선택한 행동과 함께 리플레이 메모리에 저장한다.
        # 리플레이 메모리가 일정 크기 이상으로 저장되면 매 스텝마다 학습할 수 있도록 한다.
        while not done:
            env.render()

            action = choose_action(state)

            next_state, reward, done, info = env.step(action)

            score += reward
            if not done or score == 500:
                reward = 0.1
            else:
                reward = -1

            remember(state, action, reward, next_state, done)
            if len(memory) >= train_start:
                trainModel()

            state = next_state

            if done:
                if score_avg == 0:
                    score_avg = score
                else:
                    score_avg = 0.9 * score_avg + 0.1 * score
                
                print('episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f}'.format(e, score_avg, len(memory), epsilon))
    
playGame()
