#!/usr/bin/python

import Enviroment, Policy, numpy

if __name__ == "__main__":
    try:
        reward_history = numpy.load("reward_history.npy").tolist()
    except IOError:
        reward_history = []


    #TODO: Definir as dimensoes da rede a partir do Enviroment
    policy             = Policy.ACGradientPolicy(action_dimension=5, obsv_dimension=15)
    trajectory_sampler = Enviroment.TrajectorySampler(policy=policy)

    for episode in range(0, 200):
        print("Ep %d" %episode)

        thau = trajectory_sampler.generate_trajectorys()
        r = policy.learn(thau)

        reward_history.append(r)

        print("History: %s" %reward_history)
        numpy.save("reward_history", numpy.array(reward_history))
