#!/usr/bin/python

import Enviroment, Policy

if __name__ == "__main__":
    #TODO: Definir as dimensoes da rede a partir do Enviroment
    policy             = Policy.ACGradientPolicy(action_dimension=5, obsv_dimension=15)
    trajectory_sampler = Enviroment.TrajectorySampler(policy=policy)

    for episode in range(0, 100):
        thau = trajectory_sampler.generate_trajectorys()
        policy.learn(thau)
