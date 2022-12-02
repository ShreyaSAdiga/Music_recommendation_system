from flask import Flask,render_template,request
import fer20 as fer20
import cv2
import chatbot_img as chat_img
import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/home")
def hello1():
    return render_template("home.html") 
  
@app.route("/")
def hello1_2():
    return render_template("home.html") 


@app.route("/recmnd",methods=['GET','POST'])
def hello2():
    if request.method=="POST":
        CLICK=request.form["CLICK"]
        chat_img.training()
        chat_img.chats()
        # selected=chat_img.recommend_songs()
        # res=fer20.fer123()
        # chat_img.chats()
    return render_template("recmnd_happy.html")

@app.route("/SignIn")
def hello3():
    return render_template("SignIn.html")  

@app.route("/Signup")
def hello4():
    return render_template("Signup.html")


if __name__=="__main__":
    app.run(debug=True)
