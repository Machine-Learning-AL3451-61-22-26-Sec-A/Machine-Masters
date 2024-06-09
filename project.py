!pip install textblob
!pip install keras

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import fnmatch
import pandas as pd
from textblob import TextBlob
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D, LSTM
from keras.datasets import imdb


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


@st.cache
def load_intermediate():
    files = find("adam_acc_8643",  os.getcwd())
    model = keras.models.load_model(files[0])
    return model


def model_TextBlob(text_input):
    blob = TextBlob(str(text_input))
    return (blob.sentiment.polarity + 1) / 2


def get_fixed_word_to_id_dict():
    INDEX_FROM = 3   # word index offset
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v+INDEX_FROM) for k, v in word_to_id.items()}
    word_to_id[" "] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    return word_to_id


def encode_sentence(sent):
    word_to_id = get_fixed_word_to_id_dict()
    encoded = [word_to_id[w] if w in word_to_id else 2 for w in sent.split(" ")]
    return encoded


def model_KerasIntermediate(text_input):
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

    model = load_intermediate()

    test_sentences = []
    test_sentence = str(text_input)
    test_sentence = encode_sentence(test_sentence)
    test_sentences.append(test_sentence)
    test_sentences = sequence.pad_sequences(test_sentences, maxlen=400)
    predictions = model.predict(test_sentences)
    return predictions[0]


def predictor_page():
    MODELS = {"Basic": model_TextBlob, "Intermediate": model_KerasIntermediate,
              "Complex": model_KerasIntermediate}

    option = st.selectbox('Choose an NLP Model Complexity:', list(MODELS.keys()))
    current_model = MODELS[option]

    text_label = "Text to Analyze"
    filled_text = "Enter your text here"
    ip = st.text_input(text_label, filled_text)

    show_preds = False if ((ip == filled_text) or (ip == "")) else True

    if (show_preds == True):
        positivity_scale = current_model(ip)
        result = ["Hate" if p <= 0.25 else "Demoralizing" if p <= 0.50 else "Appreciation" if p <= 0.75 else "Overwhelming"
                  for p in positivity_scale]
        result = str(result[0]) + " Speech"
        st.success('Predicted Text Class : {}'.format(result))
        latest_iteration = st.empty()
        latest_iteration.text('Measured Positivity : ' + str(positivity_scale*100) + " %")
        bar = st.progress(positivity_scale)
        st.balloons()

    st.markdown("##### Note: High Complexity/Long Text Inputs may be computationally expensive and might lead to delayed processes & performance issues.")


def about_page():
    html_temp = """
    <div style="background-color:#000000;padding:10px">
    <h2 style="color:white;text-align:center;"><b>About the Project</b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    img_top = """<br><center><img src="https://i.imgur.com/ncZTQzR.jpg" width="700px"></center>"""
    
    topic = """
    <br>
    'Text Sentiment Analysis' in 'Python' using 'Natural Language Processing (NLP)' for Negative/Positive Content Predictions.   
    Deployed on the Cloud using 'Streamlit' on the 'Heroku' Platform.
    
    The primary objective of this project is to predict the positivity in a given text and to perform classification over the 
    following speech categories :
    - Overwhelming Speech
    - Appreciation Speech
    - Demoralizing Speech
    - Hate Speech
    ---
    """
    
    topic2 = """
    The project is a part of Winter of Code and it has been initiated by <a href="https://github.com/dsc-iem">DSC-IEM</a> .
    The goal here is to use different Artificial Intelligence Algorithms for building the model and perform an in-depth exploratory analysis 
    of the obtained results. Ideas to creating a user-friendly Web-Application and deploying it to the cloud is 
    also an integral part of this Data Science Life Cycle Project. This website is an initial starter for the endless possibilities that 
    this project encloses.
    
    <p style="color:blue;">If you liked this project don't forget to star our repository😄 ! It motivates us to create great Open Source Software !</p>
     
    
    <br>
    
    [![ReadMe Card](https://github-readme-stats.vercel.app/api/pin/?username=khanfarhan10&repo=TextSentimentAnalysis&theme=dark)](https://github.com/khanfarhan10/TextSentimentAnalysis)
    """

    st.markdown(topic, unsafe_allow_html=True)
    st.markdown(img_top, unsafe_allow_html=True)
    st.markdown(topic2, unsafe_allow_html=True)


def collaborator_page():
    json_file = get_simple_contribs()
    repo = json_file
    df = pd.DataFrame()
    for each_contrib in repo:
        Total_Commits = each_contrib["total"]
        weeks = each_contrib["weeks"]
        additions = 0
        deletions = 0
        for each_week in weeks:
            additions += each_week["a"]
            deletions += each_week["d"]
        author_details = each_contrib["author"]
        author_name = author_details["login"]
        df = df.append({'name': author_name, 'commits': Total_Commits,
                        'adds': additions, 'dels': deletions}, ignore_index=True)

    df = df.astype(int, errors='ignore')
    df = df.sort_values(by=['commits', 'adds'], ascending=False)

    headings = """
        <div style="background-color:#000000;padding:10px">
        <h1 style="color:white;text-align:center;">Project Collaborators:</h1>
        </div>
        """

    html_temp = """
    <div style="background-color:#000000;padding:10px">
    <h2 style="color:white;text-align:center;"><b>Project Collaborators</b></h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    odd = True if len(df) % 2 == 1 else False

    first = """
        <html>
            <head>
                
            </head>

            <body>
                <a>"""

    mid = """
    <strong style="font-size:20px">
                        <pre class="tab">{0} <a style="font-size:14px">{1} commits </a><a style="color: #2bff00;font-size:10px">{2}++ </a><a style="color: #FF0000;font-size:10px">{3}--</a>{8}{4} <a style="font-size:14px">{5} commits </a><a style="color: #2bff00;font-size:10px">{6}++ </a><a style="color: #FF0000;font-size:10px">{7}--</a></pre>
                        <div class="github-card" data-github="{0}" data-width="350" data-height="150" data-theme="default"></div>
                        <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                        <div class="github-card" data-github="{4}" data-width="350" data-height="" data-theme="default"></div>
                        <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                    </strong> 
    """

    end = """                
            </body>
        </html>
        """

    text = """"""
    text += first

    for i in range(0, len(df)-1, 2):
        first_row = df.iloc[i, :]
        second_row = df.iloc[i+1, :]
        num_spaces = 47 - len(first_row["name"]) - len(second_row["name"]) - len(
            str(first_row["commits"])) - len(str(second_row["commits"]))
        num_spaces = num_spaces - len(str(first_row["adds"])) - \
            len(str(second_row["adds"])) - \
            len(str(first_row["dels"])) - len(str(second_row["dels"]))
        num_spaces = 9 if num_spaces < 0 else num_spaces
        spaces = " "*num_spaces
        middle = mid.format(first_row["name"], first_row["commits"], first_row["adds"], first_row["dels"],
                            second_row["name"], second_row["commits"], second_row["adds"], second_row["dels"], spaces)
        text += middle

    text += end

    if odd:
        alone = """
        <strong style="font-size:20px">
                            <pre class="tab">{0} <a style="font-size:14px">{1} commits </a><a style="color: #2bff00;font-size:10px">{2}++ </a><a style="color: #FF0000;font-size:10px">{3}--</a></pre>
                            <div class="github-card" data-github="{0}" data-width="350" data-height="150" data-theme="default"></div>
                            <script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>
                        </strong> 
        """
        last_row = df.iloc[len(df)-1, :]
        text += alone.format(last_row["name"], last_row["commits"],
                             last_row["adds"], last_row["dels"])

    components.html(text, height=700, scrolling=True, width=800)


def sidebar_nav():
    html_img = """<center><img src="https://i.imgur.com/ncZTQzR.jpg" width="300px" ></center>"""
    st.sidebar.markdown(html_img, unsafe_allow_html=True)
    st.sidebar.markdown("""## Navigation Bar: <br> """, unsafe_allow_html=True)
    current_page = st.sidebar.radio(
        " ", ["Predictions",  "Project Collaborators", "About"])

    sidetext = """
    <br><br><br><br><br>Thank you for visiting this website🤗.  
    We contribute towards open source :  
    Feel free to visit [our github repository](https://github.com/khanfarhan10/TextSentimentAnalysis)
    """
    st.sidebar.markdown(sidetext, unsafe_allow_html=True)

    all_pages = {"Predictions": predictor_page,
                 "Project Collaborators": collaborator_page, "About": about_page}

    func = all_pages[current_page]
    func()


sidebar_nav()
