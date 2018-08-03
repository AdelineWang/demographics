# This file contains functions to make post request to the consumer tables.
import requests
import json
import asyncio
import os
from graphqlclient import GraphQLClient
import sys
from eval import *
from image_emotion_gender_demo import *
from test_ethnic import *

# # Import Adelines Library.
sys.path.insert(0, '/home/ubuntu/fare/recognition/traits/src/')
#from image_emotion_gender_demo import Person_Input #<= Import line for adeline's library




# GraphQL endpoint
GraphQL_Endpoint = 'http://35.183.111.132:82/graphql'
#This function will first perform the initial addition of the new Consumer
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

    '''
    for f in listdir(path):
        if isfile(join(path, f)) and not f.startswith('.'):
            image_path_= join(path, f)
    '''        

    client = GraphQLClient(GraphQL_Endpoint)

    query = """
    mutation UpdateFacialInsights($uuid:Int!, $facialInsights:inputFaceData!){
        updateFacialInsights(uuid: $uuid, facialInsights: $facialInsights){
            uuid
        }
    }
    """
    #Obtain Facial Insights


    age, gender = get_age_gender(path)
    emotion = get_Insights(path)
    race_ = predict_ethnic(path)
    race = ETHNIC[np.argmax(race_)]

    one_insight = {'age':age, 'gender':gender, 'emotion':emotion, 'ethnicity': race}

    print(one_insight)

    '''
    variables = {
        'uuid': uuid,
        'facialInsights': facialInsights
    }
    client.execute(query, variables)
    '''

# Demo Script Call.
#addConsumer(38, 3, 3, 12345.9, "../images")

#image_path = '/Users/adelwang/Documents/Hackery/Gender-Age-Expression/GenderExpression/images/joe-biden.jpg'
#addFacialInsights("123", image_path)
