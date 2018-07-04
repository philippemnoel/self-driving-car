# This model originated from Sully Chen
# https://github.com/SullyChen/Autopilot-TensorFlow
# and was reformated and re-applied by me
import scipy.misc
import random

# coordinates
xs, ys = [], []

# points to the end of the last batch
train_batch_pointer, val_batch_pointer = 0, 0

# read data.txt (file with brake or throttle or steering angle & frame number)
with open("dataset_train_Nvidia/data.txt") as f:
    for line in f:
        xs.append("dataset_train_Nvidia/" + line.split()[0])
        # the paper by Nvidia uses the inverse of the turning radius,
        # but steering wheel angle is proportional to the inverse of turning
        # radius so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

# get number of frames & shuffle list
num_images = len(xs)
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# split dataset into training and testing sets
# training
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]
# testing
val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]
# number of frames in each set
num_train_images = len(train_xs)
num_val_images = len(val_xs)

# load training batch
def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out, y_out = [], []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

# load testing batch
def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out, y_out = [], []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
