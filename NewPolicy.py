import tensorflow as tf
import numpy as np

class Critic:
    def __init__(self, act_dimension, obsv_dimension):
        self.o = tf.placeholder(tf.float32, [None, obsv_dimension], name="CriticInput")
        a = tf.layers.dense(self.o, 20, activation=tf.tanh, name="HiddenCritic")
        self.y = tf.layers.dense(a, 1, activation=None, name="OutputCritic")

        self.Y = tf.placeholder(tf.float32, [None,], name="Example")

        self.train = tf.train.GradientDescentOptimizer(1e-5).minimize(tf.square(self.y - self.Y))

    def feedfoward(self, observations):
        observations = np.atleast_2d(observations)

        return self.y.eval(feed_dict={self.o: observations})

    def backpropagate(self, observations, returns):
        for o, r in zip(observations, returns):
            self.train.run(feed_dict={self.o:[o], self.Y:[r]})


class Actor:
    def __init__(self, act_dimension, obsv_dimension, deep_layer_dimensions=[]):
        self.layers = []
        self.o = tf.placeholder(tf.float32, shape=(None, obsv_dimension), name="ActorInput")

        a = self.o
        for l in deep_layer_dimensions:
            a = tf.layers.dense(a, l, activation=tf.nn.tanh, name="ActorHidden")

        self.y = tf.layers.dense(a, act_dimension, use_bias=False, activation=tf.nn.softmax, name="ActorOutput")

        self.advantages   = tf.placeholder(tf.float32, shape=(None,), name="Advantages")

        self.actions      = tf.placeholder(tf.int32, shape=(None,), name="Actions")

        self.actions_taken        = tf.one_hot(self.actions, act_dimension)
        self.trajectory_probs     = self.actions_taken * self.y
        self.sum_trajectory_probs = tf.log(tf.reduce_sum(self.trajectory_probs, 1))
        self.advantaged_prob      = self.sum_trajectory_probs * self.advantages
        self.sum_actor_critic     = tf.reduce_sum(self.advantaged_prob)
        self.actor_cost           = tf.negative(self.sum_actor_critic)

        self.train = tf.train.GradientDescentOptimizer(1e-4).minimize(self.actor_cost)


    def feedfoward(self, observations):
        return self.y.eval(feed_dict={self.o: observations})

    def backpropagate(self, observations, actions, advantages):
        if False:
            print("")
            print("")
            print("-- Log")
            print("- Inputs:")
            print("Observations: %s" %observations)
            print("Actions: %s" %actions)
            print("Advantages: %s" %advantages)
            print("- Loss:")
            print("Actions taken: %s" %self.actions_taken.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))
            print("Trajectory probs: %s" %self.trajectory_probs.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))
            print("Sum trajectory probs: %s" %self.sum_trajectory_probs.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))
            print("Advantaged probs: %s" %self.advantaged_prob.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))
            print("Sum actor critic: %s" %self.sum_actor_critic.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))
            print("Actor critic: %s" %self.actor_cost.eval(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages}))

        self.train.run(feed_dict={self.o:observations, self.actions:actions, self.advantages:advantages})


class Policy:
    def __init__(self, obsv_space, act_space, deep_space=[]):
        self.__gamma = 0.99
        self.__anet = Actor(act_space, obsv_space, deep_space)
        self.__cnet = Critic(act_space, obsv_space)
        saver = tf.train.Saver()

        self.sess = sess = tf.InteractiveSession()

        tf.summary.merge_all()
        tf.summary.FileWriter('tensorflowLog/', sess.graph)


        with sess.as_default():
            try:
                saver.restore(self.sess, "savedData/")

                print("All variables loaded")

            except:
                print("Previous training variables not found")
                print("Creating new set of variables")
                tf.global_variables_initializer().run()

                saver.save(self.sess,  "savedData/")


    def sample_action(self, observations):
        observations = np.atleast_2d(observations)

        pred =  self.__anet.feedfoward(observations)[0]

        return np.random.choice(len(pred), p=pred)

    def learn(self, observations, actions, rewards):
        advantages, returns = self.__advantage_estimate(observations, actions, rewards)

        observations = np.atleast_2d(observations)
        actions      = np.atleast_1d(actions)
        advantages   = np.atleast_1d(advantages)
        returns      = np.atleast_1d(returns)

        self.__cnet.backpropagate(observations, returns)
        self.__anet.backpropagate(observations, actions, advantages)


    def __advantage_estimate(self, observations, actions, rewards):
        advantages = []
        returns    = []
        G = 0

        for o, r in zip(reversed(observations), reversed(rewards)):
            G = r + self.__gamma * G
            v = self.__cnet.feedfoward(o)[0][0]

            returns.append(G)
            advantages.append(G - v)


        returns.reverse()
        advantages.reverse()

        return advantages, returns
