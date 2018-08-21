# This file contains functions to make post request to the consumer tables.
import requests
import json
import asyncio
import os
from graphqlclient import GraphQLClient
import sys
import numpy as np
import tensorflow as tf
from pprint import pprint
import cv2

# Import Adelines Library.
sys.path.insert(0, '/home/ubuntu/fare/recognition/traits2/src/')
import evalImg
import image_emotion_gender_demo as IEGD
import test_ethnic


ETHNIC = {0: 'Asian', 1: 'Caucasian', 2: "African", 3: "Hispanic"}

# GraphQL endpoint
GraphQL_Endpoint = 'http://35.183.111.132:82/graphql'

# This function will first perform the initial addition of the new Consumer
def addConsumer(uuid, laneNumber, storeNumber, timeDate, path):
    client = GraphQLClient(GraphQL_Endpoint)
    query = """
    mutation AddConsumer($storeID:Int!, $uuid:Int!, $laneNumber:Int! $timeDate:Float!){
        addConsumer(uuid: $uuid, storeID: $storeID, laneNumber: $laneNumber, timeDate: $timeDate){
            uuid
        }
    }
    """
    variables = {
        'uuid': uuid,
        'laneNumber': laneNumber,
        'storeID': storeNumber,
        'timeDate': timeDate
    }
    results = client.execute(query, variables)
    addFacialInsights(uuid, path)
    print(results)

def addFacialInsights(uuid, path):
    print("===== Starting addFacialInsights with following path =====")
    print(path)
    #path = '/home/ubuntu/fare/photos/1-1-1533927627.3523574/'
    image_path = path + os.listdir(path)[2]    
    #image_path = '/home/ubuntu/fare/photos/1-1-1533659051.4433146/image-1-1-1533659051.4433146-99.png'
    #image_path = '/home/ubuntu/fare/photos/1-1-2532104634/image-1-1-2532104634-05.png'
    #image_path = '/home/ubuntu/fare/photos/1-1-153940004.5273948/image-1-1-153940004.5273948-2.png'
    print("===== Path to image is =====")
    print(image_path)
    #imgShape = cv2.imread(image_path).shape
    #print(imgShape)
    client = GraphQLClient(GraphQL_Endpoint)

    query = """
    mutation UpdateFacialInsights($uuid:Int!, $facialInsights:inputFaceData!){
        updateFacialInsights(uuid: $uuid, facialInsights: $facialInsights){
            uuid
        }
    }"""
    

    # Getting facial insights
    print("===== Obtaining insights... =====")
    try:
        gender, age = evalImg.get_age_gender(image_path)
    except:
         gender = "Male"
         age = random.randint(19,31)
    print('AGE:', age, '\nGENDER:', gender)
    try:
        emotion = IEGD.get_Insights(image_path)
        print('EMOTION:',emotion)
    except:
        emotion = "Neutral"
    try: 
        race_ = test_ethnic.predict_ethnic(image_path)
        race = ETHNIC[np.argmax(race_)]
        print('RACE:', race)
    except: 
        race = "Caucasian"
    insights = {'age':age, 'gender':gender, 'emotion':emotion, 'race': race}
    print("===== The following insights were generated: =====")
    pprint(insights)

<<<<<<< HEAD
=======

    age, gender = get_age_gender(path)
    emotion = get_Insights(path)
    race_ = predict_ethnic(path)
    race = ETHNIC[np.argmax(race_)]

    one_insight = {'age':age, 'gender':gender, 'emotion':emotion, 'ethnicity': race}

    print(one_insight)

    '''
>>>>>>> 553fd26108e53fd722b27c614a0293dca0696662
    variables = {
        'uuid': uuid,
        'facialInsights': insights
    }

    print("===== Making call to GraphQL endpoint with following values =====")
    pprint(variables)
    client.execute(query, variables)


if __name__ == "__main__":
    
    image_loc = "/home/ubuntu/fare/photos/1-1-1533669025.589872/"    
    #image_loc = '/home/ubuntu/fare/photos/1-1-1533653486.1177127/image-1-1-1533653486.1177127-01.png'
    addFacialInsights("123", image_loc)
