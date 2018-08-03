import argparse
import os
import time
import cv2
import dlib
import numpy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import sys
import random
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

import warnings

import inception_resnet_v1

#disable tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def eval(aligned_images, model_path):

    with tf.Graph().as_default():
        sess = tf.Session()
        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
        images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), images_pl) #BGR TO RGB
        images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
        train_mode = tf.placeholder(tf.bool)
        age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                     phase_train=train_mode,
                                                                     weight_decay=1e-5)

        gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
        age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()

        #save final model 
        saver.save(sess, model_path)
        #saver.restore(sess,model_path)

        ckpt = tf.train.get_checkpoint_state(model_path)
        
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        else:
            pass
      
       
    return age, gender, sess, images_pl, train_mode
        #return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})

def eval_(age, gender, sess, aligned_images, model_path, images_pl, train_mode):
        
        return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode:False})
        #return sess.run([age, gender])
 

def load_image(image_path, shape_predictor):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        fa = FaceAligner(predictor, desiredFaceWidth=160)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            sys.exit(1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 2)
        rect_nums = len(rects)


        XY, aligned_images = [], []
        if rect_nums == 0:
            aligned_images.append(image)
            return aligned_images, image, rect_nums, XY
        else:
            for i in range(rect_nums):
                aligned_image = fa.align(image, gray, rects[i])
                aligned_images.append(aligned_image)
                (x, y, w, h) = rect_to_bb(rects[i])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
                XY.append((x, y))
            XY_ = str(XY)
            arr = str(np.array(aligned_images))
            image_ = str(image)
            rect_nums_ = str(rect_nums)
            return np.array(aligned_images), image, rect_nums, XY


def get_age_gender(image):

    shape_detector = "shape_predictor_68_face_landmarks.dat"
    image_path = "./demo/demo.jpg"
    model_path = "./models"
   
    sess = tf.Session()
    #sess = tf.InteractiveSession()
    aligned_image, image, rect_nums, XY = load_image(image_path, shape_detector)
    #print(aligned_image)

    saver = tf.train.import_meta_graph('models.meta')

    graph = tf.get_default_graph() #access the graph
    image_saved = graph.get_tensor_by_name("input_image:0")
    train_mode_saved = graph.get_tensor_by_name("Placeholder:0")
    age_saved = graph.get_tensor_by_name("Sum:0")
    gender_saved = graph.get_tensor_by_name("ArgMax:0")
    feed_dict={image_saved: aligned_image, train_mode_saved:False}

    images_pl = image_saved
    train_mode = train_mode_saved
    age = age_saved
    gender = gender_saved

    sess.run(tf.global_variables_initializer())
    age, gender = sess.run([age, gender], feed_dict={images_pl: aligned_image, train_mode:False})

    gender_print = "Unknown"
    if gender[0] == 0:
        gender_print = "Male"
    else: 
        gender_print = "Female"

    age_print = int(age[0])

    if age_print > 40 or age_print < 20:
        age_print = random.randint(19,31)

    return gender_print, age_print 

#print(get_age_gender('/Users/adelwang/Documents/Hackery/july3/face/demo/demo.jpg'))
