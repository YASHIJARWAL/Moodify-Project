import tkinter as tk
from tkinter import *

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import random
import pickle



def chat():
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    from tensorflow import keras
    import pickle
    import random
    data={"intents": [
            
            {"tag": "success",
                 "patterns": ["hey I completed my project", "I got selected", "I got promoted" ],
             "responses": ["Congrats!you deserved it","Party time!! so happy for you","So proud of you"],
             "context": [""]
            },
            {"tag": "disgusted",
                 "patterns": ["he spits while eating", "she is creepy", "he is always making me uncomfortable" ],
             "responses": ["you can talk to them about this","you are too good for h","You don't have to be around that person"],
             "context": [""]
            },
            {"tag": "angry",
                 "patterns": ["this is such a waste of time", "I'm so done now", "I am so angry about it" ],
             "responses": ["Calm down you'll be fine","It's okay I am always here","Hey champ don't waste yourself"],
             "context": [""]
            },
            {"tag": "travel",
                 "patterns": ["I wish to travel", "I want to explore myself", "I want to find as new hobby" ],
             "responses": ["Sure dude that's awesome","good to see you like this","That's a rocking idea"],
             "context": [""]
            },
            {"tag": "greet",
                 "patterns": ["Hi", "Hey", "How are you?" ],
             "responses": ["Hi good to see you","hey dude what's up?","hey I'm great how about you?"],
             "context": [""]
            },
            {"tag": "life",
                 "patterns": ["what is life?", "when will I succeed?", "Why do I fail?" ],
             "responses": ["Good things take time","It's okay, it's just a bad phase","come on you will be good"],
             "context": [""]
            },
            {"tag": "music",
            "patterns": ["what's your favorite song?", "who's your favorite artist?", "do you like jazz?" ],
            "responses": ["I like all kinds of music", "I'm not really into music, sorry", "I love classical music"],
            "context": [""]
            },
            {"tag": "movies",
            "patterns": ["what's your favorite movie?", "have you seen the latest Marvel movie?", "what's the best action movie?" ],
            "responses": ["I'm a chatbot, so I don't really watch movies", "I haven't watched any good movies lately", "I love horror movies"],
            "context": [""]
            },
            {"tag": "books",
            "patterns": ["what's your favorite book?", "have you read the latest Stephen King novel?", "what's the best self-help book?" ],
            "responses": ["I don't read books, sorry", "I haven't read any good books lately", "I love mystery novels"],
            "context": [""]
            },
            {"tag": "food",
            "patterns": ["what's your favorite cuisine?", "have you tried Indian food?", "what's the best restaurant in town?" ],
            "responses": ["I'm a chatbot, so I don't really eat food", "I haven't tried any new restaurants lately", "I love Italian food"],
            "context": [""]
            },
            {"tag": "sports",
            "patterns": ["what's your favorite sport?", "have you watched the latest soccer game?", "what's the best sports team?" ],
            "responses": ["I'm a chatbot, so I don't really watch sports", "I haven't watched any good games lately", "I love basketball"],
            "context": [""]
            },
            {"tag": "technology",
            "patterns": ["what's your favorite tech gadget?", "have you tried the latest iPhone?", "what's the best laptop for gaming?" ],
            "responses": ["I'm a chatbot, so I don't really use technology", "I haven't tried any new gadgets lately", "I love Apple products"],
            "context": [""]
            },
            {"tag": "animals",
            "patterns": ["what's your favorite animal?", "have you seen a panda before?", "what's the best pet for kids?" ],
            "responses": ["I'm a chatbot, so I don't really have a favorite animal", "I haven't seen any interesting animals lately", "I love cats"],
            "context": [""]
            },
            {"tag": "politics",
            "patterns": ["what's your political view?", "have you been following the latest election?", "who's the best politician?" ],
            "responses": ["I don't have a political view, sorry", "I haven't been following politics lately", "I don't have an opinion on politicians"],
            "context": [""]
            },
            {"tag": "nature",
            "patterns": ["what's your favorite outdoor activity?", "what's the best place to hike?" ],
            "responses": ["Being in sunshine and out is one's best part of life", "I love hiking in the mountains"],
            "context": [""]
            },
            {
            "tag": "love",
            "patterns": ["I'm in love","How do I know if I'm in love?","I can't stop thinking about them"],
            "responses": ["Love is a beautiful feeling","It sounds like you have strong feelings for them","Love is all about taking a leap of faith"],
            "context": [""]
            },
            {
            "tag": "breakup",
            "patterns": ["I just went through a breakup","How do I get over my ex?","I miss them so much"],
            "responses": ["I'm sorry to hear that","Breakups are tough but you will get through this","Take some time for yourself"],
            "context": [""]
            },
            {
            "tag": "anxiety",
            "patterns": ["I'm feeling anxious","What do I do when I have an anxiety attack?","How do I cope with anxiety?"],
            "responses": ["Take some deep breaths","Try to focus on the present moment","It's okay to ask for help"],
            "context": [""]
            },
            {
            "tag": "depression",
            "patterns": ["I'm feeling depressed","How do I deal with depression?","I don't enjoy anything anymore"],
            "responses": ["It's okay to not be okay","Depression is a common issue, but it can be treated","Have you considered seeking professional help?"],
            "context": [""]
            },
            {
            "tag": "motivation",
            "patterns": ["I need motivation","How do I stay motivated?","I'm feeling demotivated"],
            "responses": ["Think about why you started in the first place","Take small steps towards your goal","You got this!"],
            "context": [""]
            },
            {
            "tag": "stress",
            "patterns": ["I'm feeling stressed","How do I manage stress?","I can't handle the pressure"],
            "responses": ["Take a break and relax","Try some stress-relieving activities like yoga or meditation","Don't hesitate to ask for help"],
            "context": [""]
            },
            {
            "tag": "self-care",
            "patterns": ["I need to practice self-care","How do I take care of myself?","I'm neglecting my needs"],
            "responses": ["Self-care is important, make sure to prioritize it","Take some time for yourself and do something you enjoy","Remember that you deserve to take care of yourself"],
            "context": [""]
            },
            {"tag": "relationships",
            "patterns": ["How do I improve my relationship?","I'm having issues with my partner","What makes a good relationship?"],
            "responses": ["Communication is key in any relationship","Try to understand each other's perspectives","Remember to respect each other"],
            "context": [""]
            }   
            ]
    }
    # Load the intents and responses
    
    # Load the trained model
    model = keras.models.load_model('chat_model')

    # Load the tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load the label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)
    # Set the maximum length of the input sequence
    max_len = 20

    # Create the GUI window
    root = tk.Tk()
    root.geometry('2000x1000')
    root.title('Chatbot')

    # Create the chat window
    chat_window = tk.Text(root, bd=1, bg='yellow', width=50, height=8)
    chat_window.place(x=6,y=6, height=500, width=1400)

    # Create the input field
    input_field = tk.Entry(root, bg='light blue', width=50)
    input_field.place(x=128, y=600, height=35, width=265)

    # Define the function to get a response from the chatbot
    def get_response():
        # Get the user input from the input field
        user_input = input_field.get()
        input_field.delete(0, tk.END)

        # Predict the tag for the user input using the trained model
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # Select a random response from the intents dictionary based on the predicted tag
        for i in data['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
                break

        # Add the user input and chatbot response to the chat window
        chat_window.insert(tk.END, "You: " + user_input + '\n\n')
        chat_window.insert(tk.END, "Chatbot: " + response + '\n\n')

    # Create the send button
    send_button = tk.Button(root, text='Send', bg='green', activebackground='lightgreen', fg='white', font=('Arial', 12), command=get_response)
    send_button.place(x=6, y=600, height=35, width=120)

    # Start the GUI main loop
    root.mainloop()


def suggest():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    import matplotlib.pyplot as plt

    import datetime
    import os
    import cv2

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD

    from keras import regularizers
    from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
    from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
    from keras_preprocessing.image import ImageDataGenerator, load_img
    from keras.utils.vis_utils import plot_model

    from sklearn.metrics import classification_report, confusion_matrix

    main_accent_colour = "#b366ff"
    dim_colour="darkgrey"
    main_palette = ["#FBE5C0", "#DD9A30", "#F88379", "#FF6FC2", "purple", "#D086F6", "#B0D2C2", "#4C5D70", "#6FA2CE", "#382D24", "#3ACF3A", "#7D7D00"]

    import cv2

    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    import matplotlib.pyplot as plt
    import datetime
    import os
    import cv2
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from keras import regularizers
    from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
    from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
    from keras_preprocessing.image import ImageDataGenerator, load_img
    from keras.utils.vis_utils import plot_model
    from sklearn.metrics import classification_report, confusion_matrix
    from keras.models import load_model
    from time import sleep
    from keras.utils import img_to_array
    from keras.preprocessing import image
    import cv2
    import numpy as np
    
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    models = tf.keras.models.load_model('2130094')
    img_counter=0
    while True:
        _, frame = cap.read()
        labels = []
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = models.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            result=label
            img_counter += 1


    cap.release()
    cv2.destroyAllWindows()
    i=0
    result1=0
    while(i<len(emotion_labels)):
        if emotion_labels[i]==result:
            result1=i
            break
        else:
            i+=1
    print(result1)
    root = tk.Tk()
    root.title("Mood Music")
    
    def music():
            root= tk.Tk()
            root.title("Mood Music") 
            def refresh(f1):
                columns = f1.columns
                for i, column in enumerate(columns):
                    label = tk.Label(root, text=column, padx=10, pady=5)
                    label.grid(row=0, column=i, sticky='nsew')

                # Insert data into the table
                for i, row in f1.iterrows():
                    for j, column in enumerate(columns):
                        label = tk.Label(root, text=row[column], padx=10, pady=5)
                        label.grid(row=i+1, column=j, sticky='nsew')

                # Configure grid weights to allow for resizing
                for i in range(len(columns)):
                    root.grid_columnconfigure(i, weight=1)

                # Configure row weight for table data
                for i in range(len(f1) + 1):
                    root.grid_rowconfigure(i, weight=1)

                #result_label = tk.Label(root, text=f1.to_string(index=False))
            mood_music = pd.read_csv("C:/Users/KIIT/data_moods.csv")
            mood_music = mood_music[['name','artist','mood']]
            mood_music
            if(result1==0 or result1==1 or result1==2 ):
                #for angery,disgust,fear
                filter1=mood_music['mood']=='Calm'
                f1=mood_music.where(filter1)
                f1=f1.head(60).dropna()
                f1.reset_index(inplace=True)
                columns = f1.columns
                refresh(f1)
                
            if(result1==3 or result1==4):
                #for happy, neutral
                filter1=mood_music['mood']=='Happy'
                f1=mood_music.where(filter1)
                f1=f1.head(60).dropna()
                f1.reset_index(inplace=True)
                refresh(f1)

                
                #result_label.pack()
            if(result1==5):
                   #for Sad
                filter1=mood_music['mood']=='Sad'
                f1=mood_music.where(filter1)
                f1=f1.head(60).dropna()
                f1.reset_index(inplace=True)
                refresh(f1)

            if(result1==6):
                 #for surprise
                filter1=mood_music['mood']=='Energetic'
                f1=mood_music.where(filter1)
                f1=f1.head(60).dropna()
                f1.reset_index(inplace=True)
                refresh(f1)
    def link():
        import tkinter as tk
        import webbrowser
        def open_youtube_link():
            # Get song name from entry box
            song_name = entry.get()
            # Replace spaces in the song name with "+" for YouTube search query
            query = song_name.replace(" ", "+")
            # Generate YouTube link with the song name as the search query
            youtube_link = f"https://www.youtube.com/results?search_query={query}"
            # Open the YouTube link in a web browser
            webbrowser.open(youtube_link)

        # Create tkinter window
        root = tk.Tk()
        root.title("YouTube Link Generator")
        window_width = 1400
        window_height = 720
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.configure(bg="yellow")

            # Create label for input prompt
        input_label = tk.Label(root, text="Enter the name of the song:", bg="blue")
        input_label.pack()

            # Create entry box for song name input
        entry = tk.Entry(root)
        entry.pack()

            # Create button to generate and open YouTube link
        generate_button = tk.Button(root, text="Generate and Open Link", command=open_youtube_link)
        generate_button.pack()

            # Set background color for the result label
        result_label = tk.Label(root, text="", bg="yellow")
        result_label.pack()

            # Start tkinter event loop
        root.mainloop() 
    def movies():
        root = tk.Tk()
        root.title("Mood Movies")
    # Load the movie data into a pandas DataFrame
        result_label = tk.Label(root, text='', font=('Arial', 12))
        result_label.pack(pady=10)

        # Load the movie data into a pandas DataFrame
        movie = pd.read_csv('movies.csv')
        mood_movie = movie[['title', 'genres']]
                
        # Clear the contents of the label
        result_label.config(text='')
                
        # Loop over the movies in the DataFrame
        for index, row in mood_movie.iterrows():
            genres = row['genres'].split('|')
                
            # Check the movie's genres based on the user's mood
            if result1 == 0 or result1 == 1 or result1 == 2:
                if 'Documentary' in genres and 'Comedy' in genres:
        # Display the movie's title and genres in the label
                    result_label.config(text=result_label.cget('text') + f"{row['title']}\n",justify='left')

            if(result1==3 or result1==4 ):
                if 'Adventure' in genres and 'Mystery' in genres:
                    # Display the movie's title in the label
                    result_label.config(text=result_label.cget('text') + f"{row['title']}\n",justify='left')
            if(result1==5):
                if 'Action' in genres and 'Sci-Fi' in genres:
                    # Display the movie's title in the label
                    result_label.config(text=result_label.cget('text') + f"{row['title']}\n",justify='left')
            if(result1==6):
                if 'Drama' in genres and 'Romantic' in genres:
                    # Display the movie's title in the label
                    result_label.config(text=result_label.cget('text') + f"{row['title']}\n",justify='left')
                
        # Add a newline character to the label to separate the results from different moods
        result_label.config(text=result_label.cget('text') + "\n")
    r= tk.Tk()
    r.title('choice')

    # Set window size and position on the screen
    window_width = 1400
    window_height = 720
    screen_width = r.winfo_screenwidth()
    screen_height = r.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    r.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Set window background color
    r.configure(bg='white')
    button5 = tk.Button(r, text='music', width=50, height=5, command=music, bg='yellow', activebackground='orange', font=('Helvetica', 14, 'bold'))
    button5.pack(pady=5)

    # Create 'detect emotions' button with custom background colors, width, height, and font
    button6 = tk.Button(r, text='movie', width=50, height=5, command=movies, bg='aqua', activebackground='pink', font=('Helvetica', 14, 'bold'))
    button6.pack(pady=5)

    # Create 'exit' button with custom background colors, width, height, and font
    button7 = tk.Button(r, text='link', width=50, height=5, command=link, bg='magenta', activebackground='violet', font=('Helvetica', 14, 'bold'))
    button7.pack(pady=5)
    button8= tk.Button(r, text='main menu', width=50, height=5, command=r.destroy, bg='orange', activebackground='violet', font=('Helvetica', 14, 'bold'))
    button8.pack(pady=5)

    r.mainloop()

def main():
    r1 = tk.Tk()
    r1.title('MOODIFY')

    # Set window size and position on the screen
    window_width = 1400
    window_height = 720
    screen_width = r1.winfo_screenwidth()
    screen_height = r1.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    r1.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Set window background color
    r1.configure(bg='white')

    # Create 'chat' button with custom background colors, width, height, and font
    button1 = tk.Button(r1, text='Chat', width=50, height=5, command=chat, bg='yellow', activebackground='orange', font=('Helvetica', 14, 'bold'))
    button1.pack(pady=5)

    # Create 'detect emotions' button with custom background colors, width, height, and font
    button2 = tk.Button(r1, text='Detect Emotions', width=50, height=5, command=suggest, bg='aqua', activebackground='pink', font=('Helvetica', 14, 'bold'))
    button2.pack(pady=5)

    # Create 'exit' button with custom background colors, width, height, and font
    button3 = tk.Button(r1, text='Exit', width=50, height=5, command=r1.destroy, bg='red', activebackground='violet', font=('Helvetica', 14, 'bold'))
    button3.pack(pady=5)

    r1.mainloop()

main()



