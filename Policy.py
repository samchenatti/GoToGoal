import ModularNN
import numpy as np

# Classe que representa a politica. Basicamente um wrapper pra rede

class GradientPolicy:
    def __init__(self, action_dimension=1, obsv_dimension=1, deep_layers=[3, 3]):
        self.__neuralnet = ModularNN.ToyNeuralNet(layers=[obsv_dimension, 20, action_dimension], deep_activation="relu", softmax_output=True)

    def learn(self):
        self.__neuralnet.backpropagate([[ 0,  1,  1,  1,  0,  0,  1,  0, 10,  0,  0,  1,  1,  1,  1]])

    def stop(self):
        self.__neuralnet.save_weights()
        self.__neuralnet.close_session()

    def sample_action(self, o):
        o = np.array([o])
        print("Observation:")
        print(o)
        print("")

        actions_density = self.__neuralnet.feed_foward(o)

        a = np.argmax(actions_density[0])

        print("NN output")
        print(actions_density)
        print("Sum:  %f" %(actions_density.sum()))
        print("Choose action: %d" %a)

        return a
