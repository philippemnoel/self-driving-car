# Self-Driving Car Model Project

## Philippe M. Noël

This is a quick project I made in order to get more exposure to self-driving cars technology after completing the TensorFlow tutorials by Hvass-Labs. These files perform contain all you need to take datasets and reformat them so they can fit with the model used here to train a car frame-by-frame to steer, brake and accelerate (separately, based on what labels are available in your dataset. For instance the Nvidia dataset only contained steering, but the Udacity one contained acceleration and braking too). This is all inspired by this Nvidia paper: https://arxiv.org/pdf/1604.07316.pdf

I want to point out that the model is taken from Sully Chen (https://github.com/SullyChen/Autopilot-TensorFlow). He made the model that I re-used to learn myself and experiment, and I have modified a few bits to adapt it to the Udacity dataset (which he doesn't use in his GitHub), but the main part of the model was made by him and all attributions go to him. 

The datasets used can be found on my drive here: https://drive.google.com/drive/u/0/folders/1oBh4v1QxOCLP-QQlQgnLUBdfoJJhoW9T or can be found on Sully Chen's GitHub for the Nvidia dataset and on the Udacity self-driving car GitHub (https://github.com/udacity/self-driving-car/tree/master/datasets), for the Udacity datasets.

The Udacity datasets came packaged in .bag files, which are a super big pain to deal with if you are not running Linux (I was running MacOS High Sierra). The .bag files are managed by ROS and you can open them with a virtual machine or a docker container, here is the link to the installation of the latest ROS software (at time of writing this): http://wiki.ros.org/melodic/Installation.

All in all, this project was very instructive for me and while it mainly consisted of putting together work of others and assembling it to extend the Sully Chen model to another larger dataset and test the car on different roads, it did give a very interesting result and made me appreciate all the work that goes into such technology!
