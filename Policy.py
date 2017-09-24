import ModularNN
import numpy as np

# Classe que representa a politica. Basicamente um wrapper pra rede

class ACGradientPolicy:
    def __init__(self, action_dimension=1, obsv_dimension=1, deep_layers=[3, 3]):
        self.__actor_net  = ModularNN.PGNeuralNet(layers=[obsv_dimension, 20, action_dimension], deep_activation="tahn", gradient_policy=True, softmax_output=True, data_folder="SavedData/ActorData/", name="Actor", verbose=True)
        self.__critic_net = ModularNN.PGNeuralNet(layers=[obsv_dimension, 20, 1], deep_activation="tahn", gradient_policy=False, softmax_output=False, data_folder="SavedData/CriticData/", name="Critic", verbose=False)

        self.gamma = 0.009

        self.reward_history = []


    def learn(self, trajectory):
        rewards = trajectory["rewards"]
        obsv    = trajectory["observations"]
        actions = trajectory["actions"]

        r, advantages = self.__advantage_estimate(obsv, rewards)
        print("Advantages: %s" %advantages)

        self.__actor_net.backpropagate_trajectory([actions], obsv, advantages)

        return r


    def sample_action(self, o):
        o = np.array([o])
        pred = self.__actor_net.feedfoward(o)[0]

        print("Actions disttibution: %s" %pred)

        # Tomamos uma acao com base na densiade
        a = np.random.choice(len(pred), p=pred)

        print("Action taked: %s" %a)

        return a


    def __advantage_estimate(self, obsv, rewards):
        returns = []
        advantages = []
        print(obsv)
        G = totalreward = 0
        for o, r in zip(reversed(obsv), reversed(rewards)):
            returns.append(G)

            v = self.__critic_net.feedfoward([o])[0]

            advantages.append(G - v)
            G = r + self.gamma * G
            returns.reverse()
            advantages.reverse()

            totalreward += r

        # update the models
        # pmodel.partial_fit(states, actions, advantages)
        for o in obsv:
            self.__critic_net.backpropagate([o], [returns])

        return (totalreward, advantages)
