import tensorflow as tf
import numpy as np
import tf_util
import pickle
import gym

class AffinePolicy():
  """
  single layer affine network
  input -- * W -- + b -- 
  """
  def __init__(self, input_dim, output_dim):
    self.input = tf.placeholder(dtype=tf.float64, shape=(None, input_dim))
    self.targets = tf.placeholder(dtype=tf.float64, shape=(None, 1, output_dim))

    self.W = tf.get_variable("W", (input_dim, output_dim), tf.float64, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    self.b = tf.get_variable("b", (output_dim), tf.float64, tf.random_uniform_initializer(0.0, 2.0 / output_dim))
    self.output = tf.matmul(self.input, self.W) + self.b
    self.loss = tf.nn.l2_loss(self.output - self.targets)
    self.optimizer = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(self.loss)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class NueralNet():
  """
    Single hidden layer nueral net
    input -- *W1 -- +b1 -- relu -- *W2 -- +b2 -- tanh
  """
  def __init__(self, input_dim, output_dim, hidden_dims=[256]):
    self.input = tf.placeholder(dtype=tf.float64, shape=(None, input_dim))
    self.targets = tf.placeholder(dtype=tf.float64, shape=(None, 1, output_dim))

    last_dim = input_dim
    last_layer = self.input
    index = 0

    for hidden_layer_dim in hidden_dims:
      with tf.variable_scope("layer_%s_%s" % (index, hidden_layer_dim)):
        W = tf.get_variable("W", (last_dim, hidden_layer_dim), tf.float64, initializer=tf.contrib.layers.xavier_initializer())
        variable_summaries(W)
        b = tf.get_variable("b", (hidden_layer_dim), tf.float64, tf.random_uniform_initializer(0.0, 2.0 / hidden_layer_dim))
        variable_summaries(b)
        pre_activations = tf.matmul(last_layer, W) + b
        tf.summary.histogram('pre_activations', pre_activations)
        last_layer = tf.nn.tanh(pre_activations)
        tf.summary.histogram("activations", last_layer)

        last_dim = hidden_layer_dim
        index += 1

    W = tf.get_variable("Wfc", (last_dim, output_dim), tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    variable_summaries(W)
    b = tf.get_variable("bfc", (output_dim), tf.float64, tf.random_uniform_initializer(0.0, 2.0 / output_dim))
    variable_summaries(b)

    self.output = tf.matmul(last_layer, W) + b
    tf.summary.histogram("output", self.output)
    self.loss = tf.nn.l2_loss(self.output - self.targets)
    tf.summary.scalar('loss', self.loss)
    self.optimizer = tf.train.AdagradOptimizer(learning_rate=1e-4).minimize(self.loss)

def run_policy(env, policy, render=False, mean=0.0, std=1.0):
  state = env.reset()
  done = False
  reward_total = 0
  while not done:
    state -= mean
    state /= std
    output = tf_util.eval(policy.output, feed_dict={policy.input : state.reshape((1, -1))})
    state, reward, done, info = env.step(output)
    if render:
      env.render()
    reward_total += reward
  return reward_total

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_training_data', type=str)
  parser.add_argument('--train', action='store_true')
  parser.add_argument('--loadcheckpoint', type=str, default=None)
  parser.add_argument('--savecheckpoint', type=str, default=None)
  parser.add_argument('--summaries_dir', type=str, default="summaries/")
  parser.add_argument('--num_epochs', type=int, default=100)
  parser.add_argument('--minibatch_size', type=int, default=1000)
  parser.add_argument('--nueral_net', action='store_true')
  parser.add_argument('--render', action='store_true')
  args = parser.parse_args()

  env = gym.make("Humanoid-v1")

  with tf.Session() as sess:
    training_data = pickle.load(open(args.expert_training_data))
    training_observations = training_data['observations']
    training_actions = training_data['actions']

    num_examples = training_observations.shape[0]
    print "Loaded %d expert examples" % (num_examples)

    obs_dim = training_observations.shape[1]
    action_dim = training_actions.shape[2]

    print "Observation dim %s" % obs_dim
    print "Action dim %s" % action_dim

    mean = np.mean(training_observations, axis=0)
    std = np.std(training_observations, axis=0) + 1e-6

    high = env.action_space.high
    low = env.action_space.low

    policy = None
    if args.nueral_net:
      policy = NueralNet(obs_dim, action_dim, hidden_dims=[64, 64])
    else:
      policy = AffinePolicy(obs_dim, action_dim)

    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.train.SummaryWriter(args.summaries_dir,
                                      sess.graph)
    tf_util.eval(tf.global_variables_initializer())

    if args.loadcheckpoint:
      tf_util.load_state(args.loadcheckpoint)

    if args.train:
      training_iterations = args.num_epochs * num_examples / args.minibatch_size

      for t in xrange(training_iterations):
        batch_idxs = np.random.choice(np.arange(num_examples), args.minibatch_size)

        batch_actions = training_actions[batch_idxs]
        batch_observations = (training_observations[batch_idxs] - mean) / std

        feed_dict = { 
          policy.input : batch_observations, 
          policy.targets : batch_actions
        }

        loss = 0.0
        if t % 10 == 0:
          _, loss, summary = tf_util.eval([policy.optimizer, policy.loss, merged_summaries], feed_dict=feed_dict)
          summary_writer.add_summary(summary, t)
        else:  
          _, loss = tf_util.eval([policy.optimizer, policy.loss], feed_dict=feed_dict)

        if (t * args.minibatch_size) % num_examples == 0:
          epoch_number = int(t * args.minibatch_size / num_examples)
          print "Epoch: %d, loss: %s" % (epoch_number, loss / args.minibatch_size)
          reward = run_policy(env, policy, render=args.render, mean=mean, std=std)
          print "Reward: %s" % reward
          if args.savecheckpoint:
            tf_util.save_state(args.savecheckpoint)
      print "Finished training after %s epochs" % (args.num_epochs)

    trial_total_rewards = []
    for i in xrange(1000):
      if i % 100 == 0:
        print "Collecting policy performance %s..." % (i)
      trial_total_rewards.append(run_policy(env, policy))
    trial_total_rewards = np.array(trial_total_rewards)

    print "Policy results"
    print "Mean: %s, Std: %s" % (trial_total_rewards.mean(), trial_total_rewards.std())



if __name__ == "__main__":
  main()