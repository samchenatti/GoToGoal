import tensorflow as tf

class ToyNeuralNet:
    def __init__(self, layers=[2, 4, 1], gradient_policy=True, softmaxOutput=False, verbose=True, data="savedData/ReinforceNN"):
        self.layers        = layers
        self.input         = None
        self.output        = None

        self.weight        = []
        self.dc_dw         = []
        self.bias          = []
        self.dc_db         = []
        self.a             = []
        self.z             = []

        self.learning_rate = 0.1

        self.softmaxOutput   = softmaxOutput
        self.activation      = self.sigmoid
        self.data            = data
        self.gradient_policy = gradient_policy
#         self.cost          = self.likelihood_ratio()

        self.verb_mode  = verbose

        # Creates a default graph and a default session
        self.graph      = tf.Graph()
        self.sess       = None

        # Open the main session
        self.open_session()

        # Create the layers and connects them
        self.create_graphs()

        # Loads the weights
        self.load_weights()


        if self.verb_mode:
            print("If u r changing the NN shape use a new data folder!")

    def create_graphs(self):
        # There is a lot of with statements to properly define a namespace for each
        # network layer. It makes it easy to vizualise the network on the Tensorboard
        # app


        # Uses the class graph as the default
        with self.graph.as_default():

            # Build the Network using the GPU
            with tf.device("/gpu:0"):

                # Defines the input layer on its own name scope
                with tf.name_scope("Layer_0/") as scope:
                    self.a_0 = tf.placeholder(tf.float32, [1, self.layers[0]], name="Input")

                # Defines weights and biases
                self.bias.append(0)
                self.weight.append(0)
                with tf.name_scope('HiddenFullyConnected') as scope:
                    i = 1
                    for x in self.layers[1:]:
                        with tf.name_scope("Layer_%d/" %i) as scope:
                            self.bias.append(tf.Variable(tf.truncated_normal([1, x]), name="Bias"))
                            i += 1

                    i = 1
                    for x, y in zip(self.layers[:-1], self.layers[1:]):
                        with tf.name_scope("Layer_%d/" %i) as scope:
                            self.weight.append(tf.Variable(tf.truncated_normal([x, y]), name="Weight"))
                            i += 1

                print("Pesos: %d" %(len(self.weight)))


                # Defines the fowardpass graph
                self.a       = [self.a_0]
                self.z       = [0]
                for i in range(1, len(self.layers)):
                    with tf.name_scope("Layer_%d/" %i) as scope:
                        with tf.name_scope("Z_%d" %i) as scope:
                            z = tf.add(tf.matmul(self.a[i - 1], self.weight[i]), self.bias[i])

                        if i == len(self.layers) - 1 and self.softmaxOutput:
                            a = self.softmax(z)
                        else:
                            a = self.activate(z)

                        self.a.append(a)
                        self.z.append(z)

                self.a_L = self.a[-1]

                # We also need a default saver to save and load our variables
                # at each run
                with tf.name_scope("GobalSaver/") as scope:
                    self.saver = tf.train.Saver()

                self.Y = tf.placeholder(tf.float32, [1, 1], name="Input")

                # Aqui a coisa fica meio insana
                with tf.name_scope("GradientPolicy") as scope:
                    # self.likelihood_ratio = tf.log(self.a_L)
                    self.cost = tf.square(self.Y - self.a_L, name="cost")



                with tf.name_scope("Backprop/") as scope:
                    self.train_op  = tf.train.GradientDescentOptimizer(self.learning_rate)
                    self.gradients = self.train_op.compute_gradients(self.cost)


                if self.verb_mode:
                    print("Graphs created")


    # Run the feedfoward graph
    def feed_foward(self, a_0):
        with self.graph.as_default():

            r = self.a_L.eval({self.a_0: a_0}, session=self.sess)

            if self.verb_mode:
                print("IT IS ALIVE  (and thinking)  >:D")

            # tf.summary.merge_all()
            # tf.summary.FileWriter('tensorflow_log', self.sess.graph)

            return r

    def load_weights(self, sess=None):
        with self.graph.as_default():
            try:
                self.saver.restore(self.sess, self.data)

                if self.verb_mode:
                    print("All variables loaded")

            except:
                if self.verb_mode:
                    print("Previous training variables not found")
                    print("Creating new set of variables")
                self.init_variables(sess)


    def backpropagate(self, a_0, Y):
        with self.graph.as_default():
            o = self.train_op.apply_gradients(self.gradients)

            o.run(feed_dict={self.a_0: a_0, self.Y: Y}, session=self.sess)

            return self.cost.eval({self.a_0: a_0, self.Y: Y}, session=self.sess)


    def likelihood_ratio(self):
        with tf.name_scope("LikelihoodRatio/") as scope:
            lr = tf.log(self.a_L)


    def save_weights(self, sess=None):
        with self.graph.as_default():
            s =  self.saver.save(self.sess,  self.data)

            if self.verb_mode:
                print("All variables saved at %s" %s)

    def init_variables(self, sess):
        self.sess.run(tf.global_variables_initializer())

        if self.verb_mode:
            print("All variables initialized")

        self.save_weights(sess)


    def open_session(self):
        with self.graph.as_default():
            # Create the session
            self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

            if self.verb_mode:
                print("Session opened")

    def close_session(self):
        self.sess.close()

        if self.verb_mode:
            print("Session closed")


    def softmax(self, z):
        return tf.nn.softmax(z)

    def relu(self, z):
        return tf.nn.relu(z)

    def activate(self, x):
        with tf.name_scope("Activation") as scope:
            return self.activation(x)

    def sigmoid(self, x):
        # return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
        return tf.nn.sigmoid(x)

    def sigmoid_prime(x):
        return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))

    def set_activation(self, activation, activation_prime):
        return
