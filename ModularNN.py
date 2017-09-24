import tensorflow as tf

class PGNeuralNet:
    def __init__(self, layers=[2, 4, 1], gradient_policy=True, deep_activation="sigmoid", softmax_output=False, verbose=True, data_folder="savedData/", name=None):
        self.name          = name
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

        self.activation      = None
        self.softmaxOutput   = softmax_output
        self.data            = data_folder + name
        self.gradient_policy = gradient_policy

        self.verb_mode  = verbose

        # Creates a default graph and a default session
        # self.graph      = tf.Graph()
        # self.sess       = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=self.graph)
        self.graph = tf.Graph()
        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=self.graph)

        #
        self.set_activation(deep_activation)

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
                    self.a_0 = tf.placeholder(tf.float32, [None, self.layers[0]], name="Input")

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


                # Defines the fowardpass graph
                self.a       = [self.a_0]
                self.z       = [0]
                for i in range(1, len(self.layers)):
                    with tf.name_scope("Layer_%d/" %i) as scope:
                        with tf.name_scope("Z_%d" %i) as scope:
                            z = tf.add(tf.matmul(self.a[i - 1], self.weight[i]), self.bias[i])

                        if i == len(self.layers) - 1 and self.softmaxOutput:
                            a = tf.nn.softmax(z)
                        else:
                            a = self.activation(z)

                        self.a.append(a)
                        self.z.append(z)

                self.a_L = self.a[-1]

                # We also need a default saver to save and load our variables
                # at each run
                with tf.name_scope("GobalSaver/") as scope:
                    self.saver = tf.train.Saver()

                self.Y = tf.placeholder(tf.float32, [1, 1], name="Input")


                with tf.name_scope("Cost/") as scope:
                    # Aqui a coisa fica meio insana e ao mesmo tempo maravilhosa
                    # Vou explicar em portugues mesmo :P
                    if self.gradient_policy:
                        with tf.name_scope("GradientPolicy/") as scope:

                            # Lembrando que a funcao objetivo eh
                            # grad_theta( sum_upto_T( log( pi_theta( a_t | s_t) ) Â_t ) )

                            # Nos imputaremos as acoes (discretas)
                            self.actions    = tf.placeholder(tf.int32,   [None, None], name="Actions")
                            # E a vantagem para cada timestep
                            self.advantages = tf.placeholder(tf.float32, [None, None], name="Advantages" )

                            # Fazemos um one hot encode das acoes
                            self.at = actions_taken    = tf.one_hot(self.actions, self.layers[-1])

                            # A nossa rede retorna a probabilidade de todas as acoes
                            # do espaco, mas queremos considerar apenas aquelas que
                            # de fato tomamos ao longo da trajetorias.
                            # Usamos o one hot encoded como uma mascara que multiplica
                            # por zero as probabilidades das acoes que nao tomamos.
                            # O que resulta entao sao as probs pi(a_0 | s_0)...pi(a_N | s_N)
                            self.tp = trajectory_probs     = self.a_L * actions_taken

                            # Fazemos o somatorio de t = 0 ate T de log( pi( a_t | s_t) )
                            self.stp = sum_trajectory_probs = tf.log(tf.reduce_sum(trajectory_probs))

                            # Multiplicamos pela vantagem e fazemos o somatorio
                            sum_actor_critic = tf.reduce_sum(sum_trajectory_probs * self.advantages)

                            # Maximizamos a recompensa
                            self.cost  = - sum_actor_critic

                            tf.Print(self.cost, [self.cost])

                    # Funcao de custo comum
                    else:
                        self.Y    = tf.placeholder(tf.float32, [None, None], name="Y" )
                        self.cost = tf.square(self.Y - self.a_L, name="cost")


                # God bless automatic diff
                with tf.name_scope("Backpropagation/") as scope:
                    # We can also use any optimizer we want
                    self.train_op  = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

                # tf.summary.merge_all()
                # tf.summary.FileWriter('tensorflowLog/' + self.name + "/", self.sess.graph)

                if self.verb_mode:
                    print("Graphs created")


    # Run the feedfoward graph
    def feedfoward(self, a_0):
        with self.graph.as_default():

            r = self.a_L.eval({self.a_0: a_0}, session=self.sess)

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


    def backpropagate_trajectory(self, actions, observations, advantages):
        with self.graph.as_default():
            self.train_op.run(feed_dict={self.a_0: observations, self.actions: actions, self.advantages: advantages}, session=self.sess)
            # return self.tp.eval(feed_dict={self.a_0: observations, self.actions: actions, self.advantages: advantages}, session=self.sess)
            self.__save_weights()

    def backpropagate(self, a_0, Y):
        with self.graph.as_default():
            # Automatic backprop
            self.train_op.run(feed_dict={self.a_0: a_0, self.Y: Y}, session=self.sess)
            # self.save_weights()


    def __save_weights(self, sess=None):
        with self.graph.as_default():
            s =  self.saver.save(self.sess,  self.data)

            if self.verb_mode:
                print("All variables saved at %s" %s)

    def init_variables(self, sess):
        self.sess.run(tf.global_variables_initializer())

        if self.verb_mode:
            print("All variables initialized")

        self.__save_weights(sess)


    def set_activation(self, activation):
        if   activation == "sigmoid":
            self.activation = tf.nn.sigmoid

        elif activation == "relu":
            self.activation = tf.nn.relu

        elif activation == "tahn":
            self.activation = tf.nn.tanh
