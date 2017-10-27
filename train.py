#!/usr/bin/python
import vrep, NewPolicy, numpy, Enviroment
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt

if __name__ == "__main__":
    try:
        reward_history = numpy.load("reward_history.npy").tolist()
    except IOError:
        reward_history = []

    try:
        #TODO: Definir as dimensoes da rede a partir do Enviroment
        # policy             = Policy.ACGradientPolicy(action_dimension=4, obsv_dimension=16)
        policy             = NewPolicy.Policy(16, 4, [15])
        trajectory_sampler = Enviroment.TrajectorySampler(policy=policy)

        for episode in range(0, 20000000):
            print("*Ep %d" %len( reward_history ))

            observations, rewards, actions = trajectory_sampler.generate_trajectories()
            policy.learn(observations, actions, rewards)

            r = numpy.array( rewards[:-1] ).sum()

            reward_history.append(r)

            print("Episode reward: %s\nLast 10 mean: %s" %(r, numpy.mean(numpy.array( reward_history[-10:] ))))

            numpy.save("reward_history", numpy.array(reward_history))


    except KeyboardInterrupt:
        plt.plot(reward_history)
        plt.title("Rewards")
        plt.show()
