import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Load the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

# Load preprocessed data
with open('words.pkl', 'rb') as file:
    words_list = pickle.load(file)
    
with open('classes.pkl', 'rb') as file:
    classes_list = pickle.load(file)

# Load the trained model
model = load_model('chatbot_model.h5')

# Tokenize and lemmatize the input sentence
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convert a sentence into a bag of words representation
def bag_of_words(sentence):
    sentence_words = preprocess_sentence(sentence)
    bag = [0] * len(words_list)
    for w in sentence_words:
        for i, word in enumerate(words_list):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the class (intent) of the input sentence
def predict_intent(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by probability values
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # If the highest probability is below a certain threshold, return a default response
    if len(results) == 0 or results[0][1] < 0.7:
        return_list.append({'intent': 'no_intent', 'probability': '1.0'})
        return return_list
    for r in results:
        return_list.append({'intent': classes_list[r[0]], 'probability': str(r[1])})
    return return_list

# Retrieve a random response from the intents JSON file based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            break
    return response

# Example usage
'''while True:
    message = input("")
    intents = predict_intent(message)
    response = get_response(intents, intents_data)
    print(response)'''
