# DQN main
# coded by St.Watermelon

from dqn_learn import DQNagent
import gym

def main():

    max_episode_num = 500
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = DQNagent(env)

    agent.train(max_episode_num)

    agent.plot_result()

if __name__=="__main__":
    main()