#!/usr/bin/python
import ModularNN

# So um pequeno teste para testar o funcionamento do backprop

if __name__ == "__main__":
    nn = ModularNN.ToyNeuralNet()


    # 0 or 0 = 0
    # 0 or 1 = 1
    # 1 or 0 = 1
    # 1 or 1 = 1

    t = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [1]]]
    # t = [[[0, 1], [1]]]

    try:
        for f in range(0, 100000000):
            e = 0
            for i in t:
                # e += nn.backpropagate([i[0]], [i[1]])
                print("--------------------------------------------")
                print("Timestep %d" %f)
                print("Error %s" %nn.backpropagate([i[0]], [i[1]]))
                print("Input: %s" %i[0])
                print("Output: %s" %nn.feed_foward([i[0]]))
                print("--------------------------------------------")
            print(f)
            # print(e)

    except KeyboardInterrupt:
        nn.save_weights()
