# This model originated from Sully Chen
# https://github.com/SullyChen/Autopilot-TensorFlow
# and was reformated and re-applied by me
import os
import tensorflow as tf
import convolutional_model
import driving_data_preprocessing
from tensorflow.core.protobuf import saver_pb2

# initialize session
LOGDIR = './save'
sess = tf.InteractiveSession()

# hyperparameters for training
L2NormConst = 0.001
train_vars = tf.trainable_variables()
loss = tf.reduce_mean(tf.square(tf.subtract(convolutional_model.y_, convolutional_model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# save the training
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

# batch optimization, we decide on 30 epochs but could do more
epochs = 30
batch_size = 100

# train over the dataset about number of epoch times
for epoch in range(epochs):
  for i in range(int(driving_data_preprocessing.num_images / batch_size)):
    xs, ys = driving_data_preprocessing.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={convolutional_model.x: xs, convolutional_model.y_: ys, convolutional_model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data_preprocessing.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={convolutional_model.x:xs, convolutional_model.y_: ys, convolutional_model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={convolutional_model.x:xs, convolutional_model.y_: ys, convolutional_model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data_preprocessing.num_images / batch_size + i)

    # batch completed
    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

# tensorflow visualization
print("Run the command line:\n --> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
