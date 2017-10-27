import ModularNN
import numpy as np

# Classe que representa a politica. Basicamente um wrapper pra rede

class ACGradientPolicy:
    def __init__(self, action_dimension=1, obsv_dimension=1, deep_layers=None):
        self.__actor_net  = ModularNN.PGNeuralNet(layers=[obsv_dimension, 35, action_dimension], deep_activation="tahn", gradient_policy=True, softmax_output=True,
data_folder="SavedData/ActorData/", name="Actor", verbose=False, lr=1e-4, optimizer="gradient")
        self.__critic_net = ModularNN.PGNeuralNet(layers=[obsv_dimension, 25, 25, 1], deep_activation="tahn", gradient_policy=False, softmax_output=False,
data_folder="SavedData/CriticData/", name="Critic", verbose=False, lr=1e-5, optimizer="gradient")

        # Queremos que a recompensa futura importe quase tanto quanto as iniciais,
        # assim o robo aprende muito com a punicao por ter travado
        self.gamma = 0.99

        self.reward_history = []


    def learn(self, observations, actions, rewards):
        r, advantages = self.__advantage_estimate(observations, rewards, actions)
        print("PENIS")
        # print("Mine advantages: %s" %advantages)
        # print("Yours advantages: %s" %a)
        self.__actor_net.debug(actions, observations, advantages)
        self.__actor_net.backpropagate_trajectory(actions, observations, advantages)

    def sample_action(self, o):
        o = np.array([o])
        pred = self.__actor_net.feedfoward(o)[0]

        print("Actions disttibution: %s" %pred)

        # Tomamos uma acao com base na densiade
        a = np.random.choice(len(pred), p=pred)

        print("Action taked: %s" %a)

        return a

        # Actions just to print
    def __advantage_estimate(self, obsv, rewards, actions):
        returns = []
        advantages = []
        V = []

        G = totalreward = 0
        for o, r in zip(reversed(obsv), reversed(rewards)):
            v = self.__critic_net.feedfoward(o)[0][0]
            V.append(v)

            G = r + self.gamma * G
            returns.append(G)
            advantages.append(G - v)

            totalreward += r
            # print(G)

        returns.reverse()
        advantages.reverse()
        V.reverse()

        # print("V: %s" %V)
        # print("Returns: %s" %returns)
        # print("Advantages: %s" %advantages)

        for act, v, rt, adv in zip(actions, V, returns, advantages):
            print("Action: %d, V: %f, Return: %f, Advantage: %f" %(act, v, rt, adv))

        # for o, r in zip(obsv, returns):
        #     self.__critic_net.backpropagate(o, r)

        self.__critic_net.backpropagate(obsv, returns)

        return (0, advantages)