# This model originated from Sully Chen
# https://github.com/SullyChen/Autopilot-TensorFlow
# and was reformated and re-applied by me
import cv2
import subprocess
import scipy.misc
import tensorflow as tf
import convolutional_model

# initialize tensorflow session
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt") # restore previous if already trained

# steering wheel image for angle display
img = cv2.imread('img/steering_wheel.jpg',0)
rows, cols = img.shape

# vars
smoothed_angle = 0
i = 0 # iteration count

# while we still got frames to process
while (cv2.waitKey(10) != ord('q')):
    # get next frame
    full_image = scipy.misc.imread("dataset_train_Nvidia/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    # evaluate for the angle and display
    degrees = convolutional_model.y.eval(feed_dict={convolutional_model.x: [image], convolutional_model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    # make smooth angle transitions by turning the steering wheel based on the
    # difference of the current angle and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    cv2.imshow("steering wheel", cv2.warpAffine(img, M, (cols, rows)))
    i += 1 # iteration increment

# terminate
cv2.destroyAllWindows()
