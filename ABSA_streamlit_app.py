
import string
import streamlit as st
import altair as alt
import pandas as pd
import tkinter
import pickle
from textblob import TextBlob
import matplotlib
import nltk
import seaborn as sns
matplotlib.use( 'agg')
 #matplotlib.pyplot. switch backend('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import re
import string
import ftfy
import spacy
import nltk
from langid.langid import LanguageIdentifier, model
import nltk
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
import numpy as np
import pandas as pd
import nltk
#download vader from nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    df = load_data()
	
    st.sidebar.title("About")

    st.sidebar.info("This is a demo application built to analyse Mobile reviews and predict the sentiments related to the aspects hidden in the reviews. A graphical represention is done to provide a nice visual analysis")

    # page = st.sidebar. selectbox ("Choose a page", ['"Homepage", "Exploration", "Review Analysis"])
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Choose a page", ["Homepage", "Exploration", "Review Analysis"])
    # page = st.menu_items("Choose a page", ["Homepage""Exploration","Review Analysis"])
    if page =="Homepage":
        st. header ("Aspect Based Sentiment Analyser")
        st.write("As more people start sharing their reviews, more opinions are generated throughout the world for all mobile devices in ecommerce websites. Decision making is now highly dependent on all the online ratings and reviews which are available on numerous online platforms. These ratings are recorded by consumers or shoppers who have used the mobile devices and wanted to express their satisfaction or dissatisfaction and share it with the world. The reviews shared by customers have a lot of sentiments embedded in them which can provide detailed insights about the mobile device services. Moreover, these reviews cover multiple topics about the mobile devices. Identifying these topics and the associated sentiments can shed light on various aspects of the mobile device services. Since many businesses in the ecommerce industry rely on these reviews, analyzing them can be helpful in identifying areas for improvement to attract more customers. By analyzing the sentiments based on different aspects using aspect-based sentiment analysis, areas for improvement can be identified. This research aims to analyze and compare existing algorithms to understand the reviews available for mobile devices in ecommerce websites. The comparative study intends to compare the methods and express customer sentiments based on reviews on specific topics of interest, visualized in the form of graphs and dashboards.")
        st.write("Created by ; Yash Jain")

    elif page == "Exploration":
        st.title("Data Exploration" )
        user_input = st.text_input("Enter the file name", "Dataframe_combined_for_exploration.csv")
        df = pd.read_csv (user_input, encoding='latin-1')
        x_axis = st.selectbox("Choose a Prodcut name from the list", df['Product_Name'].unique(), index=3)
        #_axis = st. selectbox "Choose a variable for the y-axis", df.columns, index=4)
        visualize_data(df, x_axis)

    elif page == "Review Analysis":
        user_input = st.text_area("Enter a review", "picture clicked are amazing. Some heating problem in after long hour of use. Although i love the product, worth the money")
        analyse_data (user_input)

@st.cache_data
def load_data():
    df = pd.read_csv("Dataframe_combined_for_exploration.csv", encoding='latin-1')
    return df


def visualize_data(df, x_axis) :
	sns.set(style="whitegrid")
	ax=sns.barplot( x='Sentiment Value', y="Aspect", data=df [df['Product_Name']==x_axis])
	ax.set_xlim(-1, 1)  # Adjust the limits as needed
	for bar in ax.patches:
		if bar.get_width() < 0:  # Negative sentiment
			bar.set_color('red')
		elif bar.get_width() > 0:  # Positive sentiment
			bar.set_color('green')
	st.pyplot ()


def analyse_data(data):
    columns = ['Review','Aspect','Sentiment']
    #reviews = list(filter (None, data.strip().split('. ')))
    #reviews = [basic_data_preprocessor (review) for review in reviews]
    #reviews = [advanced_data_preprocessor (review) for review in reviews]
    reviews = preprocess_text_NA_SA(data.strip())
    #cv_model = pickle.load(open ("cv model.sav",'rb'))   ---1
    # vector = cv_model.transform(reviews)  ----2


    word_2_vec_model = pickle.load(open ('word_2_vec_model','rb'))
    review_vector = [sent_vec(sent,word_2_vec_model) for sent in reviews if len(sent)>=1]
    #review_vector =review_vector.tolist()

    if len(review_vector) >= 1:
        aspect_model = pickle.load (open ("Aspect_classifier_model",'rb'))
        sent_model = pickle.load(open ("Sentiment_supportVectorClassifier",'rb' ))

        as_pred = aspect_model.predict(review_vector)
        as_pred = [getAspectName(a) for a in as_pred]

        sent_pred_num = [getSentimentScore(text) for text in reviews]
        sent_pred = [getSentiment(s) for s in sent_pred_num]
    else:
        pass

    d = {'Review': reviews, 'Aspect':as_pred, 'Sentiment': sent_pred}
    d_plot = {'Review' :reviews, 'Aspect':as_pred, 'Sentiment': sent_pred_num}
    finaldf = pd.DataFrame (d, columns=columns)
    finaldfPlot = pd.DataFrame (d_plot, columns=columns)

    st.write(finaldf)
    sns.barplot( x='Sentiment', y="Aspect", data=finaldfPlot)
    st.pyplot ()





def preprocess_text_NA_SA(text,allowed_postags=["NOUN", "ADJ", "VERB", "PART"]):
    custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}
    custom_stopwords = custom_stopwords.union({'samsung','phone','cellphone','motorola','moto','lg','google','iphone', 'mobile', 'motto', 'pixel', 'galaxy', 'moto', 'sass'})

    # Initialize the stemmer
    #text= text.replace(',','.')
    fixed_text = ftfy.fix_text(text)
    stemmer = PorterStemmer()
    # Initialize the lemmatizer
    transformed_sentences = []
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    # Initialize the language identifier
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    # Initialize the spell checker
    spell = SpellChecker()
    for sentence in sentences:
        # Filter out non-English sentences
        lang, _ = identifier.classify(sentence)
        if lang != 'en':
            continue

        # Check and correct the spelling of words in the sentence
        corrected_words = [spell.correction(word) for word in word_tokenize(sentence) if spell.correction(word) is not None]
        # Remove special characters
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', ' '.join(corrected_words))

        # Tokenize words in the sentence
        words = word_tokenize(sentence.lower())

        # replace abbrevation with words(lol- laughing out loudly)
        #words_abbr_remov = [abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word for word in words]

        # Remove stopwords, punctuation, digits, and perform lemitizatin
        spacy_doc = nlp(" ".join(words))
        transformed_words = [token.lemma_ for token in spacy_doc
                             if token.pos_ in allowed_postags
                             and token.lemma_ not in custom_stopwords
                             and token.lemma_ not in string.punctuation
                             and not token.lemma_.isdigit()]

        # Join the transformed words into a sentence
        transformed_sentence = ' '.join(transformed_words)

        transformed_sentences.append(transformed_sentence)

    return transformed_sentences

def getVector (sent, model):
    temp = pd.DataFrame ( )
    for word in sent:
        try:
            word_vec = model[word] # if word is present in embeddings then proceed else pass
            temp = temp.append (pd. Series (word_vec), ignore_index = True)
        except:
            pass
    return temp.mean ()

from nltk.tokenize import word_tokenize

def sent_vec(sent, model_wv):
    if sent and sent.strip():     # to ignore the null sentence
        sent = word_tokenize(sent)
        vector_size = model_wv.vector_size
        wv_res = np.zeros(vector_size)
        ctr = 1
        for w in sent:
            if w in model_wv:
                ctr += 1
                wv_res += model_wv[w]
        wv_res = wv_res / ctr
        return wv_res
    else:
        return None


def getAspectName (number) :    # no changed required
    if number==1:
        return 'Camera/Display'
    elif number==0:
        return 'Battery'
    elif number==4:
        return 'Simcard/Memory'
    elif number==2:
        return 'Features/Hardware'
    elif number==3:
        return 'Money'

def getSentiment (number):        # no changed required
    if number==-1:
        return 'Negative'
    elif number==1:
        return 'Positive'

def getSentimentScore(text): # no changed required
    score = analyzer.polarity_scores(text)
    if score['compound'] > 0:
        return 1;
    else:
        return -1;

if __name__ =='__main__':
    main ()
