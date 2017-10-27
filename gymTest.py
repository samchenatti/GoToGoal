import gym
# import Policy
import NewPolicy
import numpy as np

if __name__ == "__main__":
    e = gym.make("CartPole-v0")
    # policy = Policy.ACGradientPolicy(action_dimension=2, obsv_dimension=4)
    policy  = NewPolicy.Policy(4, 2)

    reward_history = []

    for epoch in range(1, 10000):
        o = e.reset()

        actions      = []
        observations = []
        rewards      = []

        done = False
        t = 0
        while not done and t < 200:
            observations.append(o)
            a = policy.sample_action(o)

            o, r, done, err = e.step(a)

            actions.append(a)

            if done:
                r = -200

            rewards.append(r)

        policy.learn(observations, actions, rewards)

        reward_history.append(np.array(rewards[:-1]).sum())

        print("Episode %d reward: %f, Last 10 reward: %f" %(epoch, reward_history[-1], np.array(reward_history[-10:]).mean() ))

        np.save("reward_history", np.array(reward_history))
