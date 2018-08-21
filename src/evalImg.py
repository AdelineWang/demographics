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

# disable tf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

os.environ['CUDA_VISIBLE_DEVICES'] = ''
'''
def eval_two(aligned_images, model_path):
    temp = None
    print("EVAL_TWO")
    with tf.Graph().as_default():
        sess = tf.Session()
        try:
            images_pl = tf.placeholder(tf.float32, shape=[None, 480, 640, 3], name='input_image')
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
            #temp = age, gender, sess, images_pl, train_mode
            print("HERE")
        except:
	    try:
	        images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
	        print(images_pl)
		
		#images_pl = tf.reshape(images_pl, [-1, 600, 600, 600])
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
                 ckpt = tf.train.get_checkpoint_state(model_path)
                #temp = sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
            except:
                print('error occured:')
                sys.exit()
'''

def eval_two(aligned_images, model_path, shapeVal):
    print("SHAPEVAL In EVAL_TWO", shapeVal)
    temp = None
    age = 0
    gender = None
    images_pl = None
    train_mode = None
    print("SHAPEVAL", shapeVal)
    
        
    if shapeVal == 480:
        with tf.Graph().as_default():
            sess = tf.Session()
            images_pl = tf.placeholder(tf.float32, shape=[None, 480, 640, 3], name='input_image')
            #images_pl = tf.placeholder(tf.float32, shape=[None,350, 243, 3], name = 'input_image')
            print("160 SHAPE IN HERE", images_pl)
            
            #images_pl = tf.reshape(images_pl, [-1, 600, 600, 600])
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
            ckpt = tf.train.get_checkpoint_state(model_path)
            return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
    elif shapeVal == shapeVal:
        with tf.Graph().as_default():
            sess = tf.Session()
            images_pl = tf.placeholder(tf.float32, shape=[None,160, 160, 3], name='input_image')
            #images_pl = tf.placeholder(tf.float32, shape=[None,480, 640, 3], name = 'input_image')

            print("480 SHAPE IN HERE", images_pl)
            #images_pl = tf.reshape(images_pl, [-1, 600, 600, 600]
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
            ckpt = tf.train.get_checkpoint_state(model_path)
            return  sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})
    else:
        sys.exit()
        #return -1
        '''
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
        else:
            pass
        '''
        #print(sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False}))
        #return sess.run([age, gender], feed_dict={images_pl: aligned_images, train_mode: False})

def load_image_eval(image_path, shape_predictor):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor)
        fa = FaceAligner(predictor, desiredFaceWidth=160)
        try: 
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
        except:
             sys.exit()
             return None

def get_age_gender(image_path):
    print("INTO GET_AGE")
    shape_detector = "/home/ubuntu/fare/recognition/traits/trained_models/shape_predictor_68_face_landmarks.dat"
    model_path = "/home/ubuntu/fare/recognition/traits/src/models"
    '''       
    sess = tf.Session()
    #sess = tf.InteractiveSession()
    aligned_image, image, rect_nums, XY = load_image_eval(image_path, shape_detector)
    print("INTO GET_AGE 2")
    saver = tf.train.import_meta_graph('/home/ubuntu/fare/recognition/traits/src/models.meta')
    print("INTO GET_AGE 3")
    try:
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
    except:
        print("NO FACE DETECTED")
        sys.exit()
    
    gender_print = "Unknown"
    if gender[0] == 0:
        gender_print = "Male"
    else: 
        gender_print = "Female"

    age_print = int(age[0])

    if age_print > 40 or age_pri age_print = random.randint(19,31)

    return gender_print, age_print 
    '''
    print("ALIGnED_IMAGE")
    #image_path = '/home/ubuntu/fare/photos/1-1-1533940004.5273948/1533940004.5273948-2'
    image_path = '/home/ubuntu/fare/photos/1-1-1532104634/image-1-1-2532104634-03.png'
    try:  
        aligned_image, image, rect_nums, XY = load_image_eval(image_path, shape_detector)
        img = cv2.imread(image_path)
    except: 
        gender_print = "Male"
        age_print = random.randint(19,31)
    print("IMG", img)
    h,w,c = img.shape
    print("H", h)
    
    try:
        age, gender = eval_two(aligned_image, model_path, h)
        gender_print = "Unknown"
        if gender[0] == 0:
            gender_print = "Male"
        else: 
            gender_print = "Female"

        age_print = int(age[0])

        if age_print > 40 or age_print<20:
             age_print = random.randint(19,31)

    except:
        gender_print = "Male"
        age_print = random.randint(19,31)
        #sys.exit()
    #print(ages, genders)  
    


    return gender_print, age_print 
