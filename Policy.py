import ModularNN
import numpy as np

# Classe que representa a politica. Basicamente um wrapper pra rede

class ACGradientPolicy:
    def __init__(self, action_dimension=1, obsv_dimension=1, deep_layers=[3, 3]):
        self.__actor_net  = ModularNN.PGNeuralNet(layers=[obsv_dimension, 20, action_dimension], deep_activation="relu", gradient_policy=True, softmax_output=True, data_folder="SavedData/ActorData/", name="Actor")
        self.__critic_net = ModularNN.PGNeuralNet(layers=[obsv_dimension, 20, 1], deep_activation="relu", gradient_policy=False, softmax_output=False, data_folder="SavedData/CriticData/", name="Critic")

        self.gamma = 0.09

        self.reward_history = []


    def learn(self, trajectory):
        rewards = trajectory["rewards"]
        obsv    = trajectory["observations"]
        actions = trajectory["actions"]

        r, advantages = self.__advantage_estimate(obsv, rewards)

        self.__actor_net.backpropagate_trajectory([actions], obsv, advantages)


    def sample_action(self, o):
        o = np.array([o])
        pred = self.__actor_net.feedfoward(o)[0]

        print("Action dist: %s" %pred)

        # Tomamos uma acao com base na densiade
        a = np.random.choice(len(pred), p=pred)

        print("Action taken: %s" %a)

        return a


    def __advantage_estimate(self, obsv, rewards):
        returns = []
        advantages = []
        print(obsv)
        G = totalreward = 0
        for o, r in zip(reversed(obsv), reversed(rewards)):
            returns.append(G)
            advantages.append(G - self.__critic_net.feedfoward([o])[0])
            G = r + self.gamma * G
            returns.reverse()
            advantages.reverse()

            totalreward += r

        # update the models
        # pmodel.partial_fit(states, actions, advantages)
        for o in obsv:
            self.__critic_net.backpropagate([o], [returns])

        return (totalreward, advantages)
