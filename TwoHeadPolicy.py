import tensorflow as tf
import numpy as np

class ActorCriticNetwork():

    def __init__(self, layers_shape, session=None, cost=None, hidden_activation=None, output_activation=None):
        self.layers_shape  = layers_shape
        self.layers        = []
        self.cost          = cost
        self.train_op      = None
        self.learning_rate = 1e-5

        self.graph        = tf.Graph()
        self.session      = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=self.graph)

        self.__build_graphs(output_activation)

        self.__load_weights()


    def __build_graphs(self, output_activation):
        # Uses the class graph as the default
        with self.graph.as_default():

            # Build the Network using the GPU
            with tf.device("/gpu:0"):

                # Avoid mess on the tensorboard
                with tf.name_scope("UnifiedNetwork/"):

                    # Input layer
                    self.observations = tf.placeholder(tf.float32, [None, self.layers_shape[0]], name="ObservationsInput")
                    self.layers.append(self.observations)

                    # Hidden layers
                    for l in range(1, len(self.layers_shape) - 1):
                        print(l)
                        self.layers.append(tf.layers.dense(self.layers[l - 1], self.layers_shape[l], activation=tf.tanh, name="DenseLayer_%d" %l))


                with tf.name_scope("CriticHead"):
                    # Output layer
                    with tf.name_scope("UnifiedNetwork/"):
                        self.critic_output = tf.layers.dense(self.layers[-1], 1, activation=None, name="CriticOutputLayer")

                    # Example
                    self.Y = tf.placeholder(tf.float32, shape=[None,], name="Examples")

                    # Cost
                    self.critic_cost = tf.square(self.critic_output - self.Y)

                    # Gradient
                    self.critic_train_op  = tf.train.GradientDescentOptimizer(1e-5).minimize(self.critic_cost)


                with tf.name_scope("ActorHead"):
                    # Output layer
                    with tf.name_scope("UnifiedNetwork/"):
                        self.actor_output = tf.layers.dense(self.layers[-1], self.layers_shape[-1], use_bias=False, activation=tf.nn.softmax, name="ActorOutputLayer")

                    self.advantages   = tf.placeholder(tf.float32, shape=[None,], name="Advantages")

                    self.actions      = tf.placeholder(tf.int32, shape=[None,], name="Actions")

                    with tf.name_scope("Cost"):
                        self.actions_taken        = tf.one_hot(self.actions, self.layers_shape[-1])
                        self.trajectory_probs     = self.actions_taken * self.actor_output
                        self.sum_trajectory_probs = tf.log(tf.reduce_sum(self.trajectory_probs, 1), name="HERE")
                        self.sum_actor_critic     = tf.reduce_sum(self.sum_trajectory_probs * self.advantages)
                        self.actor_cost           = tf.negative(self.sum_actor_critic)

                    self.actor_train_op  =  tf.train.GradientDescentOptimizer(1e-5).minimize(self.actor_cost)

                # self.session.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()

                tf.summary.merge_all()
                tf.summary.FileWriter('tensorflowLog/', self.session.graph)

    def feedfoward_critic(self, observations):
        observations = np.atleast_2d(observations)

        return self.critic_output.eval(feed_dict={self.observations:observations}, session=self.session)

    def backpropagate_critic(self, observations, values):
        observations = np.atleast_2d(observations)
        values       = np.atleast_1d(values)

        self.critic_train_op.run(feed_dict={self.observations: observations, self.Y: values}, session=self.session)

    def feedfoward_actor(self, observations):
        observations = np.atleast_2d(observations)

        return self.actor_output.eval(feed_dict={self.observations: observations}, session=self.session)

    def backpropagate_both(self, observations, actions, advantages, returns):
        observations = np.atleast_2d(observations)
        actions      = np.atleast_1d(actions)
        advantages   = np.atleast_1d(advantages)
        returns      = np.atleast_1d(returns)

        if False:
            print("- Debug")
            print("observations: %s" %observations)
            print("actions: %s" %actions)
            print("advantages: %s" %advantages)
            print("returns: %s" %returns)

            print("Ouput %s" %self.actor_output.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print(self.actions_taken.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print(self.trajectory_probs.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print("Traj probs %s" %self.sum_trajectory_probs.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print("Sum ac %s" %self.sum_actor_critic.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print(self.actor_cost.eval(feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions, self.Y: returns}, session=self.session))
            print("\n\n")

        # self.critic_train_op.run(feed_dict={self.observations:observations, self.Y: returns}, session=self.session)
        self.actor_train_op.run (feed_dict={self.observations:observations, self.advantages:advantages, self.actions:actions}, session=self.session)


    def __load_weights(self):
        self.verb_mode = True

        with self.session.graph.as_default():
            try:
                self.saver.restore(self.sess, "savedData")

                if self.verb_mode:
                    print("All variables loaded")

            except:
                if self.verb_mode:
                    print("Previous training variables not found")
                    print("Creating new set of variables")
                self.session.run(tf.global_variables_initializer())
                self.__save_weights()

    def __save_weights(self, sess=None):
        with self.session.graph.as_default():
            s =  self.saver.save(self.session,  "savedData")

            if self.verb_mode:
                print("All variables saved at %s" %s)



class Policy:
    def __init__(self, obsv_space, act_space):
        self.__gamma = 0.99
        self.__acnet = ActorCriticNetwork([obsv_space, 55, 55, 55, 55, act_space])

    def sample_action(self, observations):
        pred =  self.__acnet.feedfoward_actor(observations)
        print(pred)
        pred = pred[0]
        return np.random.choice(len(pred), p=pred)

    def learn(self, observations, actions, rewards):
        # advantages, returns = self.__advantage_estimate(observations, actions, rewards)

        self.__acnet.backpropagate_both(observations, actions, rewards, [])


    def __advantage_estimate(self, observations, actions, rewards):
        advantages = []
        returns    = []
        G = 0

        for o, r in zip(reversed(observations), reversed(rewards)):
            G = r + self.__gamma * G
            v = self.__acnet.feedfoward_critic(o)[0][0]

            advantages.append(G - v)
            returns.append(G)

        returns.reverse()
        advantages.reverse()

        return advantages, returns
