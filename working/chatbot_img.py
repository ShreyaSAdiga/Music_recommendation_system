import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

'''

def chats():
    with open(r'intents.json') as file:
        data = json.load(file)

    training_sentences = []
    training_labels = []
    labels = []
    responses = []


    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    #model.summary()

    epochs = 700
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("chat_model")

    import pickle

    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)



    import json 
    import numpy as np
    from tensorflow import keras
    from sklearn.preprocessing import LabelEncoder

    import colorama 
    colorama.init()
    from colorama import Fore, Style, Back

    import random
    import pickle

    import cv2
    from fer import FER
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from time import sleep 

    def getEmotion():

        video=cv2.VideoCapture(0)

        a=0

        while True:
            a=a+1
            check, frame = video.read()
            sleep(1)
            cv2.imshow("Capturing",frame)

            if cv2.waitKey(1) & 0xFF==ord('q') or a==5:
                break

        showPic=cv2.imwrite("Photo.jpg",frame)


        video.release()
        cv2.destroyAllWindows

        input_image=cv2.imread('Photo.jpg')
        """
        try:
          filename = take_photo()
          print('Saved to {}'.format(filename))
    
          # Show the image which was just taken.
          # display(Image(filename))
        except Exception as err:
          # Errors will be thrown if the user does not have a webcam or if they do not
          # grant the page permission to access it.
          print(str(err))
        input_image=Image(filename)"""

        emotion_detector = FER(mtcnn=True)

        result=emotion_detector.detect_emotions(input_image)

        bounding_box=result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(input_image,(
          bounding_box[0], bounding_box[1]),(
          bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255), 2,)

        emotion_name, score = emotion_detector.top_emotion(input_image)
        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (211, 211,211) if score < 0.01 else (255, 0, 0)
            emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))

            max_key = max(emotions,key=emotions.get)
            cv2.putText(input_image,emotion_score,
        			(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
        			cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
            # print(emotion_name)
            print(emotion_score)
        #Save the result in new image file
        cv2.imwrite("emotion.jpg", input_image)

        # Read image file using matplotlib's image module
        result_image = mpimg.imread('emotion.jpg')
        imgplot = plt.imshow(result_image)
        # Display Output Image
        plt.show()
        print(max_key)

    with open(r"\intents.json") as file:
        data = json.load(file)


    def chat():
        # load trained model
        model = keras.models.load_model('chat_model')

        # load tokenizer object
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20

        while True:
            print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
            inp = input()
            if inp.lower() == "pose":
                getEmotion()
                break

            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                 truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])

            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))

            # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

    print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
    # chat()
'''
import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getEmotion():
    video=cv2.VideoCapture(0)

    a=0

    while True:
        sleep(1)
        a=a+1
        check, frame = video.read()
        cv2.imshow("Capturing",frame)

        if cv2.waitKey(1) & 0xFF==ord('q') or a==5:
            break

    showPic=cv2.imwrite("Photo.jpg",frame)


    video.release()
    cv2.destroyAllWindows

    input_image=cv2.imread('Photo.jpg')

    emotion_detector = FER(mtcnn=True)

    result=emotion_detector.detect_emotions(input_image)

    bounding_box=result[0]["box"]
    emotions = result[0]["emotions"]
    cv2.rectangle(input_image,(
      bounding_box[0], bounding_box[1]),(
      bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255), 2,)

    emotion_name, score = emotion_detector.top_emotion(input_image)
    for index, (emotion_name, score) in enumerate(emotions.items()):
        color = (211, 211,211) if score < 0.01 else (255, 0, 0)
        emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))

        cv2.putText(input_image,emotion_score,
    			(bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
    			cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
        # print(emotion_name)
        print(emotion_score)
    print(type(emotions))
    max_key = max(emotions,key=emotions.get)
    #Save the result in new image file
    cv2.imwrite("emotion.jpg", input_image)
    '''
    # Read image file using matplotlib's image module
    result_image = mpimg.imread('emotion.jpg')
    imgplot = plt.imshow(result_image)
    # Display Output Image
    plt.show()'''
    # max_key = max(emotion_score, key=emotion_score.get)
    return max_key
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama 
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle
import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep 

def training():
    with open(r'intents.json') as file:
        data = json.load(file)

    training_sentences = []
    training_labels = []
    labels = []
    responses = []


    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])

    #model.summary()

    epochs = 700
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("chat_model")

    import pickle

    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


def recommend_songs(emotion):
    songs={
        'happy':['happy1','happy2','happy3','happy4','happy5','happy6','happy7','happy8','happy9','happy10'],
        'calm':['calm1','calm2','calm3','calm4','calm5','calm6','calm7','calm8','calm9','calm10'],
    }
    '''
    if emotion=='happy':
        # random.seed(100)
        # abc=random.sample(range(1, 10), 1)
        selected_song=songs[emotion][random.randint(0, 9)]
    elif emotion=='calm':
        selected_song=songs[emotion][random.randint(0, 9)]'''
    if emotion=='happy' or emotion=='sad' or emotion=='neutral':
        emotion123='happy'
    else:
        emotion123='calm'
    selected_song=songs[emotion123][random.randint(0, 9)]
    return selected_song

def chats():
    with open(r'intents.json') as file:
        data = json.load(file)

    # load trained model
    model = keras.models.load_model('chat_model')
    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    # parameters
    max_len = 20
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "pose":
            emotion123=getEmotion()
            rec_songs=recommend_songs(emotion123)
            return (rec_songs)
            # rec_songs123=song_recommendation(rec_songs)
            # print(rec_songs123)
            break
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))




