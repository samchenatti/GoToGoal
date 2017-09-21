import ModularNN
import numpy as np

# Classe que representa a politica. Basicamente um wrapper pra rede

class GradientPolicy:
    def __init__(self, action_dimension=1, obsv_dimension=1, deep_layers=[3, 3]):
        self.__neuralnet = ModularNN.ToyNeuralNet(layers=[obsv_dimension, 3, 5, 3, action_dimension])

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
        print("Choose action: %d" %a)

        return a
