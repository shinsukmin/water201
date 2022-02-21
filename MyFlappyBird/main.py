# DeepLearningFlappyBird 코드를 내임맛에 맞게 수정해보자.
# 
# 
# 
# 
import tensorflow as tf
import sys

from DQN import DQN
sys.path.append("game/")
import numpy as np

print(np.__version__)

a = DQN()

print(type(a.s))