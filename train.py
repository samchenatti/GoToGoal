#!/usr/bin/python

import Enviroment, Policy

if __name__ == "__main__":
    policy             = Policy.GradientPolicy(action_dimension=7, obsv_dimension=15)
    trajectory_sampler = Enviroment.TrajectorySampler(policy=policy)

    thau = trajectory_sampler.generate_trajectorys()
