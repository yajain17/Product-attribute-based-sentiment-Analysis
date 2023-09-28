#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from langdetect import detect
import string
from textblob import TextBlob

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from langid.langid import LanguageIdentifier, model
from spellchecker import SpellChecker
import ftfy
import nltk
from nltk.stem import WordNetLemmatizer


import pandas as pd
import glob


# 1. I wish to keep 'Not'
# 2. Replace ',' with '.' so setence tokenization can happen
# 3. Filtering out non english sentence.
# 4. Removing special characters from sentence
# 5. Lowercase sentence
# 6. First tokenize sengtence into words
# 7. 1.Remove stopwords and 2.punctuation and 3.Keeping only english words 4.Stemming
# 8. spellchecker
# 

# In[2]:


# Path of the folder containing CSV files
path = r'Topic_modeling_Project/'

# Reading all the CSV files
filenames = glob.glob(path + "/*.csv")
print('File names:', filenames)

# Initializing an empty DataFrame
final_csv = pd.DataFrame()

# Iterating over each CSV file
for file in filenames:
    print(file)
    # Reading each CSV file and appending to the final DataFrame
    df = pd.read_csv(file, error_bad_lines=False)  # Add error_bad_lines parameter
    final_csv = final_csv.append(df)

# Printing the combined DataFrame
print('Final DataFrame:')
print(final_csv)


# In[3]:


final_csv.rename(columns={'Produktnamen':'Product_Name','Kundenbewertung':'User_Rating','Titel':'Review_title','Inhalt':'Review','Anzahl_hilfreich':'Review_voting'},inplace=True)
final_csv.columns


# In[4]:


df= final_csv[['Product_Name','ASIN','User_Rating','Review_title','Review','Review_voting']]
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[ ]:





# In[7]:


df


# In[116]:


df = df.drop_duplicates()
df.info()


# In[31]:


df.to_csv('combineDataset.csv', index=True)


# In[8]:


df=pd.read_csv('combineDataset.csv')
df.info()


# In[9]:


random_samples = df.sample(n=50, random_state=42)
random_samples.head()


# In[10]:


abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "glhf" : "good luck have fun",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "prime minister",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}


# In[13]:


import nltk
nltk.download('punkt')
get_ipython().system('pip install nltk')
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

custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_NA(text,allowed_postags=["NOUN", "ADJ"]):
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
        words_abbr_remov = [abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word for word in words]

        # Remove stopwords, punctuation, digits, and perform lemitizatin 
        spacy_doc = nlp(" ".join(words_abbr_remov))
        transformed_words = [token.lemma_ for token in spacy_doc
                             if token.pos_ in allowed_postags
                             and token.lemma_ not in custom_stopwords
                             and token.lemma_ not in string.punctuation
                             and not token.lemma_.isdigit()]

        # Join the transformed words into a sentence
        transformed_sentence = ' '.join(transformed_words)

        transformed_sentences.append(transformed_sentence)

    return transformed_sentences


# Example usage



# In[15]:


preprocess_text_NA('mobile is not good. Dont buy')


# In[126]:


from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import re
import string
import ftfy
import spacy

nlp = spacy.load("en_core_web_sm")

custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_try(text, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    # Initialize the stemmer
    stemmer = PorterStemmer()
    # Initialize the lemmatizer
    transformed_sentences = []

    # Tokenize into sentences
    sentences = sent_tokenize(text)

    for sentence in sentences:
        # Remove special characters
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

        # Tokenize words in the sentence
        words = word_tokenize(sentence.lower())

        # Remove stopwords, punctuation, digits, and perform lemmatization
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


# Example usage
text = 'This is a great Samsung Galaxy A54 cell phone that is budget-friendly. The Galaxy A54, is thicker than my current and previous cellphones like the s20 Fe galaxy. The screen and picture quality is okay, but the camera resolution could be better. The audio quality is great when you watch videos or listen to music. I think that for a budget phone, the battery life is impressive to speak and navigate without playing many games. And it comes with a USB-C charger with no wall block.'

transformed_sentences = preprocess_text(text)

print(transformed_sentences)


# In[13]:


random_samples.info()


# In[19]:


df= pd.read_csv('combineDataset.csv')


# In[20]:


import numpy as np
df_reset = df.reset_index(drop=True)
df_parts = np.array_split(df_reset, 140)


# In[21]:


transformed_dfs_letmit_NA =[]


# In[ ]:


import time

# Create an empty list to store the transformed DataFrames

start_time_main = time.time()
for i in range(0,140):
    print(i)
    start_time = time.time()

    # Code or process you want to measure the execution time of
    transformed_sentences = df_parts[i]['Review'].apply(preprocess_text_NA)

    # Create a new DataFrame with the transformed sentences
    transformed_df = pd.DataFrame({'Transformed': transformed_sentences})

    # Save the transformed DataFrame to the list
    transformed_dfs_letmit_NA.append(transformed_df)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")
    print("----------completed")
end_time_main = time.time()
elapsed_time_main = end_time_main - start_time_main
print("elapsed_time_main:", elapsed_time_main, "seconds")


# In[25]:


len(transformed_dfs_letmit_NA)


# In[27]:


df_combined4 = pd.concat(transformed_dfs_letmit_NA)

type(df_combined4)
df_Review_sentence4  = df_combined4.explode('Transformed')
len(df_Review_sentence4)


# In[28]:


df_Review_sentence4.to_csv('transformed_dfs_letmit_NA.csv', index= True )


# In[71]:


df_Review_sentence4


# In[70]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# function to plot most frequent terms
def freq_words(x, terms=30):
    all_words = ' '.join(x)
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 'terms' most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.show()

# Call the function
freq_words(df_Review_sentence4['Transformed'])


# In[29]:


import spacy
from spacy.lang.en import English
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



# In[9]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[92]:


custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_title(text,allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    # Initialize the stemmer
    fixed_text = ftfy.fix_text(text)
    stemmer = PorterStemmer()
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    transformed_sentence = ''

    # Initialize the language identifier
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    # Initialize the spell checker
    spell = SpellChecker()

    # Filter out non-English sentences
    lang, _ = identifier.classify(text)
    if lang == 'en':
        # Check and correct the spelling of words in the sentence
        corrected_words = [spell.correction(word) for word in word_tokenize(text) if spell.correction(word) is not None]
        
        # Remove special characters
        transformed_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', ' '.join(corrected_words))

        # Tokenize words in the sentence
        words = word_tokenize(transformed_sentence.lower())
        
        # replace abbreviation with words (lol- laughing out loudly)
        words_abbr_remov = [abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word for word in words]

        # Remove stopwords, punctuation, digits, and perform lemmatization 
        spacy_doc = nlp(" ".join(words_abbr_remov))
        transformed_words = [token.lemma_ for token in spacy_doc
                             if token.pos_ in allowed_postags
                             and token.lemma_ not in custom_stopwords
                             and token.lemma_ not in string.punctuation
                             and not token.lemma_.isdigit()]

        # Join the transformed words into a sentence
        transformed_sentence = ' '.join(transformed_words)

    return transformed_sentence


# In[94]:


import time

# Create an empty list to store the transformed DataFrames

start_time_main = time.time()
for i in range(70, 140):
    print(i)
    start_time = time.time()

    # Code or process you want to measure the execution time of
    transformed_sentences = df_parts[i]['Review_title'].apply(preprocess_text_title)

    # Create a new DataFrame with the transformed sentences
    transformed_df = pd.DataFrame({'Transformed_title': transformed_sentences})

    # Save the transformed DataFrame to the list
    transformed_dfs_title.append(transformed_df)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")
    print("----------completed")
end_time_main = time.time()
elapsed_time_main = end_time_main - start_time_main
print("elapsed_time_main:", elapsed_time_main, "seconds")


# In[91]:


transformed_dfs_title=[]


# In[87]:


df_parts[0]


# In[89]:


df_Review_sentence


# In[95]:


df_combined_title = pd.concat(transformed_dfs_title)

type(df_combined_title)

len(df_combined_title)


# In[47]:


df_combined_title


# In[103]:


df_combined_title.to_csv('df_Review_title.csv', index= True )


# In[35]:


df_combined_title = pd.read_csv('df_Review_title.csv')


# In[36]:


df_combined_title


# In[37]:


# Perform left join based on the index values
result2 = df.merge(df_combined_title, how='left', left_index=True, right_index=True)

# Output the result


# In[38]:


result2.head(20)


# In[39]:


Df_combined4 = result2.merge(df_Review_sentence4, how='left', left_index=True, right_index=True)


# In[40]:


Df_combined4.head(100)


# In[41]:


len(Df_combined4)


# In[72]:


# Filtering Transformed to have atleast few words in each rows


filtered_df4 = Df_combined4[Df_combined4['Transformed'].str.split().str.len() > 2]


filtered_df4 = filtered_df4.drop_duplicates()


# In[73]:


filtered_df4


# In[44]:


filtered_df4.info()


# In[74]:


filtered_df4['User_Rating'] = filtered_df4['User_Rating'].str.split().str[0].astype(float)
filtered_df4['Review_voting'] = filtered_df4['Review_voting'].str.split().str[0]

filtered_df4['Review_voting'] = filtered_df4['Review_voting'].replace('one', 1)
filtered_df4['Review_voting'] = filtered_df4['Review_voting'].replace('One', 1)

filtered_df4['Review_voting'] = filtered_df4['Review_voting'].str.replace(',', '')
filtered_df4['Review_voting'].fillna(0, inplace=True)
filtered_df4['Review_voting'] = filtered_df4['Review_voting'].astype(int)


# In[75]:


filtered_df4.info()


# In[76]:


filtered_df4 = filtered_df4.drop_duplicates()
filtered_df4.info()


# In[51]:


filtered_df4.to_csv('filtered_df4_Noun_adj.csv', index=True)


# In[49]:


filtered_df4


# In[77]:


filtered_df4.rename(columns={'Transformed':'Review_sentence'},inplace=True)
filtered_df4


# In[83]:


filtered_df5=filtered_df4.copy()


# # most frequent words, To clean the garbage

# In[86]:


filtered_df5['Review_sentence'] = filtered_df4['Review_sentence'].str.replace(r'\bsassing\b|\biphone\b|\bmobile\b|\bmotto\b|\bpixel\b|\bgalaxy\b|\bmoto\b', 'phone', regex=True)
filtered_df5['Review_sentence'] = filtered_df5['Review_sentence'].str.replace(r'\bphone\b', '', regex=True)
filtered_df5['Review_sentence'] = filtered_df5['Review_sentence'].str.replace(r'\b(price|money|cost)\b', 'price', regex=True)



# In[87]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# function to plot most frequent terms
def freq_words(x, terms=75):
    all_words = ' '.join(x)
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 'terms' most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.show()

# Call the function
freq_words(filtered_df5['Review_sentence'])


# In[88]:


filtered_df5['Review_sentence'].info()


# # Least frequent words, To clean the garbage

# In[154]:


from collections import Counter

# Tokenize the sentences into words
words = ' '.join(filtered_df5['Review_sentence']).split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the words that occur only once
unique_words = [word for word, count in word_counts.items() if count == 1]

len(unique_words)


# In[153]:


# Remove the unique words from each sentence
filtered_sentences = []
for sentence in filtered_df5['Review_sentence']:
    filtered_sentence = ' '.join([word for word in sentence.split() if word not in unique_words])
    filtered_sentences.append(filtered_sentence)

# Update the 'Review_sentence' column with the filtered sentences
filtered_df5['Review_sentence'] = filtered_sentences


# In[547]:


from collections import Counter

# Tokenize the sentences into words
processed_docs = ' '.join(filtered_df5['Review_sentence']).split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the frequency of 'good' and 'great'
frequency_good = word_counts.get('no', 0)
frequency_great = word_counts.get('not', 0)

print("Frequency of 'good':", frequency_good)
print("Frequency of 'great':", frequency_great)


# In[161]:


filtered_df5['Review_sentence']


# In[177]:


# Print the DataFrame
filtered_df5['Review_sentence'].to_csv('filtered_df5_review.csv', index=True)


# # Words occuring less than 5

# In[178]:


from collections import Counter

# Tokenize the sentences into words
words = ' '.join(filtered_df5['Review_sentence']).split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the words that occur less than 5 times
rare_words = [word for word, count in word_counts.items() if count < 5]

# Remove the rare words from the sentences
filtered_sentences = []
for sentence in filtered_df5['Review_sentence']:
    filtered_sentence = ' '.join([word for word in sentence.split() if word not in rare_words])
    filtered_sentences.append(filtered_sentence)

# Update the 'Review_sentence' column in the DataFrame
filtered_df5['Review_sentence'] = filtered_sentences


# In[180]:


rare_words


# In[179]:


filtered_df5['Review_sentence'].to_csv('filtered_df5_review_filtered.csv', index=True)


# In[181]:


filtered_df5 = filtered_df5[filtered_df5['Review_sentence'].apply(lambda x: len(x.split()) >= 2)]


# In[182]:


filtered_df5.info()


# In[176]:


# Print the DataFrame
df_sentences.to_csv('df_sentences.csv', index=True)


# In[281]:


from collections import Counter

# Tokenize the sentences into words
words = ' '.join(filtered_df5['Review_sentence']).split()

# Calculate the frequency of each word
word_counts = Counter(words)

# Specify the words of interest
target_words = [
    'application',
    'call',
    'text',
    'message',
    'email',
    'incoming',
    'music',
    'notification',
    'voice',
    'outgoing',
    'setting',
    'icon',
    'permission',
    'navigation',
    'finger',
    'face',
    'print',
    'facial',
    'recognition',
    'back',
    'unlock',
    'scanner',
    'mask',
    'sensor',
    'fingerprint'
]

# Sort the target words by frequency in descending order
target_words_sorted = sorted(target_words, key=lambda word: word_counts.get(word, 0), reverse=True)

# Print the frequency of each target word in descending order
for word in target_words_sorted:
    frequency = word_counts.get(word, 0)
    print(word)


# In[214]:


filtered_df5['Review_sentence'] = filtered_df5['Review_sentence'].str.replace('great', 'good')


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from gensim import matutils
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import scipy
# Import the necessary libraries
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.corpora import Dictionary




# In[105]:


from sklearn.feature_extraction.text import CountVectorizer


# In[264]:


# Step 2: Prepare the Document-Term Matrix 
vectorizer = CountVectorizer(ngram_range=(1, 2),stop_words='english')
document_term_matrix = vectorizer.fit_transform(filtered_df5['Review_sentence'])
Feature_name= vectorizer.get_feature_names_out()
df_bow5 = pd.DataFrame(document_term_matrix.toarray(),columns= Feature_name)
 


# In[259]:


# Sum the occurrences of each bigram across all documents
bigram_counts = df_bow5.sum(axis=0)

# Sort the bigrams by their frequencies in descending order
sorted_bigrams = bigram_counts.sort_values(ascending=False)

# Get the top 75 most frequent bigrams
top_75_bigrams = sorted_bigrams.head(75)

# Print the top 75 most frequent bigrams
top_75_bigrams = pd.DataFrame(top_75_bigrams)
top_75_bigrams.head(75)
top_75_bigrams.to_csv('top_75_bigrams.csv') 


# In[262]:


from collections import Counter
import re

# Concatenate all sentences into a single string
all_sentences = ' '.join(filtered_df5['Review_sentence'])

# Tokenize the string into individual words
words = re.findall(r'\b\w+\b', all_sentences)

# Create a counter to count the occurrences of each word
word_counter = Counter(words)

# Get the most common 200 words and their counts
most_common_words = word_counter.most_common(200)

# Print the most common words and their counts
for word, count in most_common_words:
    print(word, count)

most_common_words = pd.DataFrame(most_common_words)

most_common_words.to_csv('most_common_words_noun_adj.csv') 


# In[308]:


corpusn5


# In[353]:


import gensim
from gensim.models import CoherenceModel
corpusn5 = matutils.Sparse2Corpus(scipy.sparse.csc_matrix(df_bow5.transpose()))
id2wordn5 = dict((v, k) for k, v in vectorizer.vocabulary_.items())

lda_model5 = gensim.models.ldamodel.LdaModel(corpus=corpusn5,
                                           id2word=id2wordn5,
                                           num_topics=5,passes=5,iterations=300)

# Print the Keyword in the 10 topics
print(lda_model5.print_topics())

# Import the necessary libraries
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.corpora import Dictionary

# Create the gensim Dictionary object
dictionary3 = Dictionary.from_corpus(corpusn5, id2wordn5)

# Enable notebook mode for pyLDAvis
pyLDAvis.enable_notebook()

# Prepare the visualization
vis = gensimvis.prepare(lda_model5, corpusn5, dictionary3)

# Display the visualization
pyLDAvis.display(vis)


# # Guided LDA

# In[334]:


camera = ['camera', 'picture', 'photo', 'video','quality','click','zoom','pixel','camera good','good camera','shot','excellent','clarity','pro','awesome','sharp','amazing']
simcard = ['sim', 'card', 'sim card', 'micro', 'insert', 'slot', 'tray', 'connect', 'dual','activate', 'swap','network','service'  ]
battery = ['battery', 'life',  'time' ,'usage','charge','charger','day','use','full','last','heavy','battery life','heavy use','whole day','night','average','moderate','short','hot']
money = ['price', 'value', 'worth','range','budget','reasonable','deal','good price','great price', 'good value','work great','excellent','experience']
features = ['application','call','fingerprint','speaker','finger','music','gaming','cpu','sensor','setting','text','message','recognition','music','notification','scanner','face','voice','email','unlock','navigation','facial','icon','fingerprint reader','fingerprint sensor', 'fingerprint scanner']



# In[335]:


token_vectorizer = CountVectorizer( min_df=10, stop_words=stop_words, ngram_range=(1, 2))
X = token_vectorizer.fit_transform(filtered_df5['Review_sentence'])


# In[ ]:


token_vectorizer


# In[471]:


#self added 

Feature_name= token_vectorizer.get_feature_names_out()
df_bow_GLDA = pd.DataFrame(X.toarray(),columns= Feature_name)
df_bow_GLDA

corpus_GLDA = matutils.Sparse2Corpus(scipy.sparse.csc_matrix(df_bow_GLDA.transpose()))

id2word_GLDA = dict((v, k) for k, v in token_vectorizer.vocabulary_.items())


# In[472]:


from gensim.models import guidedlda


# In[473]:


tf_feature_names = token_vectorizer.get_feature_names()


# In[474]:


word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))


# In[475]:


# Removing prior words that are not part of vocabulary

camera = [x for x in camera if x in list(word2id.keys())]
simcard = [x for x in simcard if x in list(word2id.keys())]
battery = [x for x in battery if x in list(word2id.keys())]
money = [x for x in money if x in list(word2id.keys())]
features = [x for x in features if x in list(word2id.keys())]


# In[476]:


seed_topic_list = [
    camera,
    simcard,
    battery,
    money,
    features
]


# In[477]:


corpus_GLDA


# In[478]:


model = guidedlda.GuidedLDA(n_topics=5, random_state=20)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id
model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)


# In[479]:


n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[480]:


n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[481]:


dics = {'topic 0': 'Camera/Display',
'topic 1':'simcard/Memory',
'topic 2':'Battery',
'topic 3':'Money',
'topic 4':'Features/Hardware'
}


# In[482]:


doc_topic = model.transform(X)
print(doc_topic)


# In[483]:


columns_label = ['topic {}'.format(i) for i in range(5)]  # number of topics
topic_vector = pd.DataFrame(doc_topic, columns = columns_label)#dataframe of doc-topics
print(topic_vector.shape)
topic_vector.round(2).head(10)


# In[484]:


topic_vector = topic_vector.round(2)
topic_vector


# In[485]:


filtered_df5 =  filtered_df5['Review_sentence'].reset_index()


# In[535]:


Df_final = filtered_df5.merge(topic_vector, how='left', left_index=True, right_index=True)


# In[536]:


Df_final = Df_final.rename(columns=dics)
Df_final.to_csv('lets_check_aspect2.csv',index =True )


# In[537]:


Df_final.info()


# In[538]:


Df_final['Aspect'] = Df_final[['Camera/Display', 'simcard/Memory', 'Battery', 'Money', 'Features/Hardware']].idxmax(axis=1)
Df_final['Max_value'] = Df_final[['Camera/Display', 'simcard/Memory', 'Battery', 'Money', 'Features/Hardware']].max(axis=1)

# Apply condition to 'Aspect' and 'Max_value' columns

Df_final.loc[Df_final['Max_value'] < 0.65, 'Aspect'] = None


# In[539]:


Df_final.to_csv('Df_Topic_modeling_new2.csv',index =True )


# In[494]:


model


# In[495]:


panel = pyLDAvis.sklearn.prepare(model, X, token_vectorizer, mds='tsne')
panel


# In[497]:


pyLDAvis.save_html(panel, 'GuidedLDA_Final_topics')


# In[540]:


Df_final[['Camera/Display', 'simcard/Memory', 'Battery', 'Money', 'Features/Hardware']]


# In[541]:


Df_final['Aspect']


# In[601]:


df=Df_final
df


# In[602]:


df = df[df['Aspect'].notnull()]


# In[603]:


df


# In[546]:


Df_final.to_csv('Topic_Modelling_complete',index =True )


# # Sentiment Analysis 

# In[34]:


# Path of the folder containing CSV files
path = r'Topic_modeling_Project/'

# Reading all the CSV files
filenames = glob.glob(path + "/*.csv")
print('File names:', filenames)

# Initializing an empty DataFrame
final_csv = pd.DataFrame()

# Iterating over each CSV file
for file in filenames:
    print(file)
    # Reading each CSV file and appending to the final DataFrame
    df = pd.read_csv(file, error_bad_lines=False)  # Add error_bad_lines parameter
    final_csv = final_csv.append(df)

# Printing the combined DataFrame
print('Final DataFrame:')
print(final_csv)


final_csv.rename(columns={'Produktnamen':'Product_Name','Kundenbewertung':'User_Rating','Titel':'Review_title','Inhalt':'Review','Anzahl_hilfreich':'Review_voting'},inplace=True)

df= final_csv[['Product_Name','ASIN','User_Rating','Review_title','Review','Review_voting']]



df.dropna(subset=['Review', 'Review_title'], inplace=True)


df = df.drop_duplicates()
df.shape
df.info()
df.head()


# In[16]:


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

custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_NA_SA(text,allowed_postags=["NOUN", "ADJ", "VERB", "PART"]):
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



# In[163]:


preprocess_text_NA_SA("The phone works just fine. My only issue is HUGE. I'm in my car a bunch and NEED my music. Can't get the Bluetooth to connect for ANYTHING!!! üò°")


# In[22]:


preprocess_text_NA_SA("camera is not good . Major issue with charging as well")


# In[28]:


import numpy as np
df_reset = df.reset_index(drop=True)
df_parts = np.array_split(df_reset, 140)


# In[ ]:





# In[29]:


len(df_parts)


# In[31]:





# In[ ]:


import time

# Create an empty list to store the transformed DataFrames

start_time_main = time.time()
for i in range(0,140):
    print(i)
    start_time = time.time()

    # Code or process you want to measure the execution time of
    transformed_sentences = df_parts[i]['Review'].apply(preprocess_text_NA_SA)

    # Create a new DataFrame with the transformed sentences
    transformed_df = pd.DataFrame({'Transformed_Sentences': transformed_sentences})

    # Save the transformed DataFrame to the list
    transformed_dfs_NA_SA.append(transformed_df)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Elapsed time:", elapsed_time, "seconds")
    print("----------completed")
end_time_main = time.time()
elapsed_time_main = end_time_main - start_time_main
print("elapsed_time_main:", elapsed_time_main, "seconds")
df_Review_sentence.to_csv('df_Review_sentence_Cleaned_data_main_24th_1stfile.csv',index =True )


# In[35]:


len(transformed_dfs_NA_SA)


# In[36]:


df_combined = pd.concat(transformed_dfs_NA_SA )

type(df_combined)
df_Review_sentence  = df_combined.explode('Transformed_Sentences')
len(df_Review_sentence)

df_combined


# In[40]:


df_Review_sentence


# In[98]:


df_combined = pd.concat(transformed_dfs_NA_SA )

type(df_combined)
df_Review_sentence  = df_combined.explode('Transformed_Sentences')
len(df_Review_sentence)


# In[99]:


df_Review_sentence


# In[100]:


df_Review_sentence.dropna(subset=['Transformed_Sentences'], inplace=True)
df_Review_sentence.info()


# In[338]:


df_Review_sentence


# In[101]:


df_Review_sentence = df_Review_sentence.drop_duplicates()
df_Review_sentence.info()

df_Review_sentence


# In[102]:


# Filtering Transformed to have atleast few words in each rows


df_Review_sentence = df_Review_sentence[df_Review_sentence['Transformed_Sentences'].str.split().str.len() >= 2]
df_Review_sentence.info()


# In[42]:


df_Review_sentence.info()
df_Review_sentence.to_csv('df_Review_sentence_Cleaned_data_final_24th_2ndfile.csv',index =True )


# # data cleaning done, lets do sentiment Analysis.

# In[63]:


import numpy as np
import pandas as pd
import nltk
#download vader from nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#creating an object of sentiment intensity analyzer
sia= SentimentIntensityAnalyzer()
#uploading csv file

#Polarity_scores: This function returns the sentiment strength based on the given input statement/text.


#Let us now create a new column in our CSV file that stores the polarity scores of each review.



# In[65]:


x = sia.polarity_scores ('Camera is not good')

df_Review_sentence.loc[df_Review_sentence.compound>0,'Sentiment']='Positive'
df_Review_sentence.loc[df_Review_sentence.compound==0,'Sentiment']='Neutral'
df_Review_sentence.loc[df_Review_sentence.compound<0,'Sentiment']='Negative'


# In[44]:


#creating new column scores using polarity scores function
df_Review_sentence['scores']=df_Review_sentence['Transformed_Sentences'].apply(lambda body: sia.polarity_scores(str(body)))
df_Review_sentence.head()
#Similarly, we then create three different columns each for compound scores, positive scores, and negative scores.





# In[ ]:





# In[105]:


df_Review_sentence['compound']=df_Review_sentence['scores'].apply(lambda score_dict:score_dict['compound'])

df_Review_sentence['pos']=df_Review_sentence['scores'].apply(lambda pos_dict:pos_dict['pos'])

df_Review_sentence['neg']=df_Review_sentence['scores'].apply(lambda neg_dict:neg_dict['neg'])

df_Review_sentence['Sentiment']=''
df_Review_sentence.loc[df_Review_sentence.compound>0,'Sentiment']='Positive'
df_Review_sentence.loc[df_Review_sentence.compound==0,'Sentiment']='Neutral'
df_Review_sentence.loc[df_Review_sentence.compound<0,'Sentiment']='Negative'


# In[45]:


df_Review_sentence =  df_Review_sentence.reset_index()



# # combing this with maindataset.
# 
# 

# In[128]:


df_Review_sentence
df_Review_sentence.to_csv('df_Review_sentence_20june_main.csv',index =True )


# In[127]:


df_Review_sentence =pd.read_csv('df_Review_sentence_20june_main.csv')


# In[128]:


df_Review_sentence.info()


# In[129]:


df.info()


# In[49]:


df


# In[130]:


df_Sentiment_analysis_20june = df.merge(df_Review_sentence, how='left', left_index=True, right_on='index_column')


# In[131]:


df


# In[132]:


df_Sentiment_analysis_20june 


# # topic modelling LDA

# In[133]:


df_Review_sentence.rename(columns={'Transformed_Sentences':'Review_sentence'},inplace=True)
df_Review_sentence


# In[134]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# function to plot most frequent terms
def freq_words(x, terms=75):
    all_words = ' '.join(x)
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 'terms' most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.show()

# Call the function
freq_words(df_Review_sentence['Review_sentence'])


# In[135]:


from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

# function to plot most frequent terms
def freq_words(x, terms=75):
    all_words = ' '.join(x)
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 'terms' most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.show()

# Call the function
freq_words(df_Review_sentence['Review_sentence'])


# In[136]:


df_Review_sentence['Review_sentence'] = df_Review_sentence['Review_sentence'].str.replace(r'\bsassing\b|\biphone\b|\bmobile\b|\bmotto\b|\bpixel\b|\bgalaxy\b|\bmoto\b|\bsass\b', 'phone', regex=True)
df_Review_sentence['Review_sentence'] = df_Review_sentence['Review_sentence'].str.replace(r'\bphone\b', '', regex=True)


# In[137]:


from collections import Counter

# Tokenize the sentences into words
words = ' '.join(df_Review_sentence['Review_sentence']).split()

# Count the frequency of each word
word_counts = Counter(words)

# Get the words that occur only once
unique_words = [word for word, count in word_counts.items() if count <=5 ]

len(unique_words)


# In[138]:


# Remove the unique words from each sentence
filtered_sentences = []
for sentence in df_Review_sentence['Review_sentence']:
    filtered_sentence = ' '.join([word for word in sentence.split() if word not in unique_words])
    filtered_sentences.append(filtered_sentence)

# Update the 'Review_sentence' column with the filtered sentences
df_Review_sentence['Review_sentence'] = filtered_sentences


# # LDA

# In[139]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from gensim import matutils
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import scipy
# Import the necessary libraries
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.corpora import Dictionary




# In[140]:


from sklearn.feature_extraction.text import CountVectorizer


# In[141]:


# Step 2: Prepare the Document-Term Matrix 
vectorizer = CountVectorizer(ngram_range=(1, 2),stop_words='english')
document_term_matrix = vectorizer.fit_transform(df_Review_sentence['Review_sentence'])
Feature_name= vectorizer.get_feature_names_out()
df_bow5 = pd.DataFrame(document_term_matrix.toarray(),columns= Feature_name)
 


# In[142]:


import gensim
from gensim import matutils
import scipy

from gensim.models import CoherenceModel
corpusn5 = matutils.Sparse2Corpus(scipy.sparse.csc_matrix(df_bow5.transpose()))
id2wordn5 = dict((v, k) for k, v in vectorizer.vocabulary_.items())

lda_model5 = gensim.models.ldamodel.LdaModel(corpus=corpusn5,
                                           id2word=id2wordn5,
                                           num_topics=9,passes=5,iterations=300)

# Print the Keyword in the 10 topics
print(lda_model5.print_topics())



# In[62]:


# Import the necessary libraries
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.corpora import Dictionary

# Create the gensim Dictionary object
dictionary3 = Dictionary.from_corpus(corpusn5, id2wordn5)

# Enable notebook mode for pyLDAvis
pyLDAvis.enable_notebook()

# Prepare the visualization
vis = gensimvis.prepare(lda_model5, corpusn5, dictionary3)

# Display the visualization
pyLDAvis.display(vis)


# In[40]:


pyLDAvis.save_html(vis, 'LDA_Final_topics_17th_july',LDA_Final_topics_17th_july.html)


# # GUIDED LDA

# In[143]:


camera = ['camera', 'picture', 'photo', 'video','quality','click','zoom','camera good','good camera','shot','excellent','clarity','pro','awesome','sharp','amazing']
simcard = ['sim', 'card', 'sim card', 'micro', 'insert', 'slot', 'tray', 'connect', 'dual','activate', 'swap','network','service'  ]
battery = ['battery', 'life',  'time' ,'usage','charge','charger','day','use','full','last','heavy','battery life','heavy use','whole day','night','average','moderate','short','hot']
money = ['price', 'value', 'worth','range','budget','reasonable','deal','good price','great price', 'good value','work great','excellent','experience']
features = ['application','call','fingerprint','speaker','finger','music','gaming','cpu','sensor','setting','text','message','recognition','music','notification','scanner','face','voice','email','unlock','navigation','facial','icon','fingerprint reader','fingerprint sensor', 'fingerprint scanner']



# In[144]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

token_vectorizer = CountVectorizer( min_df=10, stop_words=stop_words, ngram_range=(1, 2))
X = token_vectorizer.fit_transform(df_Review_sentence['Review_sentence'])


# In[145]:


#self added 

Feature_name= token_vectorizer.get_feature_names_out()
df_bow_GLDA = pd.DataFrame(X.toarray(),columns= Feature_name)
df_bow_GLDA

corpus_GLDA = matutils.Sparse2Corpus(scipy.sparse.csc_matrix(df_bow_GLDA.transpose()))

id2word_GLDA = dict((v, k) for k, v in token_vectorizer.vocabulary_.items())


# In[146]:


from gensim.models import guidedlda
tf_feature_names = token_vectorizer.get_feature_names()

word2id = dict((v, idx) for idx, v in enumerate(tf_feature_names))




# In[147]:


# Removing prior words that are not part of vocabulary

camera = [x for x in camera if x in list(word2id.keys())]
simcard = [x for x in simcard if x in list(word2id.keys())]
battery = [x for x in battery if x in list(word2id.keys())]
money = [x for x in money if x in list(word2id.keys())]
features = [x for x in features if x in list(word2id.keys())]


# In[148]:


seed_topic_list = [
    camera,
    simcard,
    battery,
    money,
    features
]


# In[149]:


corpus_GLDA


# In[150]:


model = guidedlda.GuidedLDA(n_topics=5, random_state=20)
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id
model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)


# In[151]:


import numpy as np
n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[152]:


n_top_words = 20
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
     topic_words = np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))


# In[153]:


dics = {'topic 0': 'Camera/Display',
'topic 1':'simcard/Memory',
'topic 2':'Battery',
'topic 3':'Money',
'topic 4':'Features/Hardware'
}


# In[ ]:





# In[154]:


doc_topic = model.transform(X)
print(doc_topic)


# In[155]:


columns_label = ['topic {}'.format(i) for i in range(5)]  # number of topics
topic_vector = pd.DataFrame(doc_topic, columns = columns_label)#dataframe of doc-topics
print(topic_vector.shape)
topic_vector.round(2).head(10)


# In[156]:


topic_vector = topic_vector.round(2)
topic_vector


# In[157]:


import pyLDAvis.sklearn
panel = pyLDAvis.sklearn.prepare(model, X, token_vectorizer, mds='tsne')
panel


# In[56]:


pyLDAvis.save_html(panel, 'GuidedLDA_Final_topics_17thjuly',GuidedLDA_Final_topics_17thjuly.html)


# In[ ]:


df_Review_sentence


# In[127]:


df_Review_sentiment = df_Review_sentence.merge(topic_vector, how='left', left_index=True, right_index=True)


# In[128]:


df_Review_sentiment = df_Review_sentiment.rename(columns=dics)
df_Review_sentiment.to_csv('Sentiment_labelANalysis_24th_june.csv',index =True )
df_Review_sentiment 


# In[185]:


df_Review_sentiment = df_Review_sentiment.rename(columns=dics)
df_Review_sentiment.to_csv('Sentiment_labelANalysis_20th_june.csv',index =True )
df_Review_sentiment 


# In[129]:


df_Review_sentiment['Aspect'] = df_Review_sentiment[['Camera/Display', 'simcard/Memory', 'Battery', 'Money', 'Features/Hardware']].idxmax(axis=1)
df_Review_sentiment['Max_value'] = df_Review_sentiment[['Camera/Display', 'simcard/Memory', 'Battery', 'Money', 'Features/Hardware']].max(axis=1)




# In[130]:


df_Review_sentiment


# In[131]:


# Apply condition to 'Aspect' and 'Max_value' columns

df_Review_sentiment.loc[df_Review_sentiment['Max_value'] < 0.65, 'Aspect'] = None


# In[132]:


df_Review_sentiment


# In[205]:


df_Review_sentiment = df_Review_sentiment[df_Review_sentiment['Review_sentence'].notnull()]


# In[206]:


df_Review_sentiment.isnull().sum()


# In[82]:


df_Review_sentiment.to_csv('Almost_final_dataset_24th_june.csv',index =True )


# In[72]:


df_Review_sentiment = pd.read_csv('Almost_final_dataset_20th_june.csv' )
df_Review_sentiment      [df_Review_sentiment['index_column']==2]


# In[120]:


df_Review_sentiment.loc[(df_Review_sentiment['index_column'] == 2) & (df_Review_sentiment['Sentiment'] != 'Neutral'), ['index_column', 'Review_sentence', 'Sentiment', 'Aspect', 'Max_value']]


# In[121]:


df.iloc[2]['Review']


# In[126]:


df_Review_sentiment.loc[(df_Review_sentiment['index_column'] == 0) &( df_Review_sentiment['Max_value'] >0.7), ['index_column', 'Review_sentence', 'Sentiment', 'Aspect', 'Max_value']]


# # Aspect modelling

# # ASPECT CLASSIFICATION - MODELLING 

# 1.word2vec

# In[186]:


import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string


# In[187]:


import gensim.downloader as api
print(list(gensim.downloader.info()['models'].keys()))


# In[188]:


wv = api.load('word2vec-google-news-300')


# In[189]:


wv


# In[153]:


with open('word_2_vec_model','wb') as file:
    pickle.dump(wv,file)


# In[154]:


with open('word_2_vec_model','rb') as file:
    mp = pickle.load(file)
    
x = mp["hello"]


# In[155]:


type(x)


# In[207]:


from nltk.tokenize import sent_tokenize

def sent_vec(sent):
    sent = word_tokenize(sent)
    #print(sent)
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res/ctr
    return wv_res


# In[208]:


l = sent_vec(" hello how are you")
type(l)


# In[209]:


df_Review_sentiment['Review_sentence'].isnull().sum()


# In[195]:


df_Review_sentiment['Review_sentence']


# In[210]:


df_Review_sentiment['Word2vec'] = df_Review_sentiment['Review_sentence'].apply(sent_vec)


# In[211]:


df_Review_sentiment['Word2vec']


# In[212]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_Review_sentiment['Aspect_target'] = encoder.fit_transform(df_Review_sentiment['Aspect'])


# In[213]:


X = df_Review_sentiment['Word2vec'].to_list()
y = df_Review_sentiment['Aspect_target'].to_list()


# In[214]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)


# In[216]:


from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression(multi_class='ovr',max_iter=1000 , C= 10.0, penalty = 'l1', solver= 'liblinear' )
classifier_1.fit(X_train,y_train)



# In[218]:


from sklearn import metrics
predicted = classifier_1.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("Logistic Regression F1 Score:", f1)


# In[219]:


import pickle


# In[253]:


with open('Aspect_classifier_model','wb') as file:
    pickle.dump(classifier_1,file)


# In[ ]:





# In[ ]:





# #### Logistic Regression F1 Score: 0.8706271395303261
# 

# In[226]:


df_Review_sentiment


# ## DO i need to fine tune or pretune word2vec on my DATA? lets check

# In[227]:


wv.key_to_index


# In[228]:


# Initialize an empty set to store the unique tokens
vocab = set()

# Tokenize each sentence and update the vocabulary
for sentence in df_Review_sentiment['Review_sentence']:
    tokens = word_tokenize(sentence)
    vocab.update(tokens)

# Print the vocabulary
print("Vocabulary:" vocab)


# In[229]:


not_avail=0
avail=0
for i in vocab:
    if i not in wv.key_to_index:
        not_avail+=1
    else:
        avail+=1
print(not_avail)
print(avail)        


# In[230]:


len(vocab)


# In[232]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Assuming you have X_train and y_train data available

# Create a logistic regression classifier
classifier = LogisticRegression(multi_class='ovr')

# Define the parameter grid
parameters = [
    {'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}]
 

# Create an instance of GridSearchCV
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring= 'f1_macro',cv = 5,
                           verbose=0)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Access the best estimator and best parameters
best_classifier = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best parameters
print("Best Parameters:", best_params)


# In[233]:


best_params


# ### TESTING

# In[239]:


predicted = classifier.predict(sent_vec('worth penny').reshape(1,300))
predicted


# # TESTING OTHER MODELS

# In[241]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[242]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    #'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[243]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred, average='weighted')
    recall = metrics.recall_score(y_test,y_pred, average='weighted')
    f1 = metrics.f1_score(y_test,y_pred, average='weighted')
    

    
    return accuracy,precision,recall,f1


# In[244]:


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for name,clf in clfs.items():
    
    current_accuracy,current_precision,current_recall,current_f1 = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Recall - ",current_recall)
    print("f1 - ",current_f1)
    print("*****************************************************")
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)
    f1_scores.append(current_f1)


# In[245]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'f1':f1_scores}).sort_values('f1',ascending=False)
performance_df


# # Sentiment Classification

# ## 1.TFIDF

# In[301]:


df_Review_sentiment=df_Review_sentiment[df_Review_sentiment['Sentiment']!= 'Neutral']
df_Review_sentiment


# In[248]:


df_Review_sentiment.shape


# In[319]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_Review_sentiment['Sentiment_target'] = encoder.fit_transform(df_Review_sentiment['Sentiment'])



# In[320]:


df_Review_sentiment


# In[264]:


df_Review_sentiment.head()


# In[303]:


df_Review_sentiment


# In[345]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500, stop_words='english', min_df=8)


# In[346]:


X= df_Review_sentiment['Review_sentence'].values
y = df_Review_sentiment['Sentiment_target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


# In[349]:


svc = SVC(kernel='sigmoid', gamma=1.0)
classifier_svc = svc.fit(X_train,y_train)
from sklearn import metrics
predicted = classifier.predict(X_test)
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("Logistic Regression F1 Score:", f1)


# In[350]:


with open('Sentiment_supportVectorClassifier','wb') as file:
    pickle.dump(classifier_svc,file)


# In[329]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[330]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    #'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred, average='weighted')
    recall = metrics.recall_score(y_test,y_pred, average='weighted')
    f1 = metrics.f1_score(y_test,y_pred, average='weighted')
    

    
    return accuracy,precision,recall,f1


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for name,clf in clfs.items():
    
    current_accuracy,current_precision,current_recall,current_f1 = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Recall - ",current_recall)
    print("f1 - ",current_f1)
    print("*****************************************************")
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)
    f1_scores.append(current_f1)


# In[314]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('lrc', lrc), ('rfc', rfc)],voting='soft')
voting.fit(X_train,y_train)

predicted = voting.predict(X_test)
print(" Accuracy:",metrics.accuracy_score(y_test, predicted))
print(" Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print(" Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("F1 Score:", f1)


# In[315]:


voting = VotingClassifier(estimators=[('svm', svc), ('lrc', lrc), ('rfc', rfc)],voting='soft')


# In[316]:


voting.fit(X_train,y_train)


# In[317]:


predicted = voting.predict(X_test)
print(" Accuracy:",metrics.accuracy_score(y_test, predicted))
print(" Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print(" Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("F1 Score:", f1)


# In[318]:


# Applying stacking
estimators=[('svm', svc), ('lrc', lrc), ('ETC', etc)]
final_estimator=RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train,y_train)

predicted = clf.predict(X_test)
print(" Accuracy:",metrics.accuracy_score(y_test, predicted))
print(" Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print(" Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("F1 Score:", f1)


# In[ ]:





# In[ ]:





# In[300]:


from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10], 
              'gamma': [1, 0.1, 0.01]
              } 
  
grid = GridSearchCV(SVC(),  param_grid, refit = True, cv=3 ,verbose = 3, scoring='f1_macro')
  
# fitting the model for grid search
grid.fit(X_train,y_train)


# # ******************************

# # 2. word2vec

# In[1]:


df_Review_sentiment


# In[322]:


X = df_Review_sentiment['Word2vec'].to_list()
y = df_Review_sentiment['Sentiment_target'].to_list()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[323]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    #'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}



# In[326]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred, average='weighted')
    recall = metrics.recall_score(y_test,y_pred, average='weighted')
    f1 = metrics.f1_score(y_test,y_pred, average='weighted')
    

    
    return accuracy,precision,recall,f1


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for name,clf in clfs.items():
    
    current_accuracy,current_precision,current_recall,current_f1 = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Recall - ",current_recall)
    print("f1 - ",current_f1)
    print("*****************************************************")
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)
    f1_scores.append(current_f1)


# In[324]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('svm', svc), ('lrc', lrc), ('rfc', rfc)],voting='soft')
voting.fit(X_train,y_train)

predicted = voting.predict(X_test)
print(" Accuracy:",metrics.accuracy_score(y_test, predicted))
print(" Precision:",metrics.precision_score(y_test, predicted, average='weighted'))
print(" Recall:",metrics.recall_score(y_test, predicted, average='weighted'))

f1 = metrics.f1_score(y_test, predicted, average='weighted')
print("F1 Score:", f1)


# In[ ]:





# In[ ]:


# 3 countvectorizer


# In[294]:


cv = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=8)


# In[295]:


X= df_Review_sentiment['Review_sentence'].values
y = df_Review_sentiment['Sentiment_target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


# In[296]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    #'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred, average='weighted')
    recall = metrics.recall_score(y_test,y_pred, average='weighted')
    f1 = metrics.f1_score(y_test,y_pred, average='weighted')
    

    
    return accuracy,precision,recall,f1


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for name,clf in clfs.items():
    
    current_accuracy,current_precision,current_recall,current_f1 = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    print("Recall - ",current_recall)
    print("f1 - ",current_f1)
    print("*****************************************************")
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    recall_scores.append(current_recall)
    f1_scores.append(current_f1)


# In[ ]:





# In[135]:


preprocess_text_NA('camera is not good')


# In[130]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Preprocess text
def preprocess_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Handle negations
    modified_tokens = []
    negation = False
    negation_words = set(["not", "no", "n't", "never", "none"])
    
    for word in tokens:
        if word in negation_words:
            negation = not negation
        elif negation:
            modified_tokens.append("not_" + word)
        else:
            modified_tokens.append(word)
    
    # Join the preprocessed words back into a sentence
    preprocessed_text = ' '.join(modified_tokens)
    
    return preprocessed_text



# In[ ]:


# Example usage
text = "Camera is not good."
preprocessed_text = preprocess_text(text)

# Perform sentiment analysis on preprocessed text
sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(preprocessed_text)

# Print sentiment scores
print(sentiment_scores)


# In[138]:


preprocess_text('camera was not good. samsung camera is never good . not liked the battery')


# In[139]:


preprocess_text_NA('camera was not good. samsung camera is never good . not liked the battery')


# In[145]:


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

custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_NA_try(text,allowed_postags=["NOUN", "ADJ"]):
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
        
        # replace abbreviation with words (e.g., "lol" with "laughing out loudly")
        abbreviations = {'lol': 'laughing out loudly'}  # Add your abbreviation dictionary
        words_abbr_removed = [abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word for word in words]

        # Remove stopwords, punctuation, digits, and perform lemmatization 
        spacy_doc = nlp(" ".join(words_abbr_removed))
        transformed_words = [token.lemma_ for token in spacy_doc
                             if token.pos_ in allowed_postags
                             and token.lemma_ not in custom_stopwords
                             and token.lemma_ not in string.punctuation
                             and not token.lemma_.isdigit()]

        # Join the transformed words into a sentence
        transformed_sentence = ' '.join(transformed_words)

        transformed_sentences.append(transformed_sentence)

    # Join the transformed sentences into a single string
    transformed_text = ' '.join(transformed_sentences)

    return transformed_text

# Example usage
text = "never purchasing this phone again."
preprocessed_text = preprocess_text_NA_try(text)

print(preprocessed_text)


# In[148]:


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

custom_stopwords = set(stopwords.words('english')) - {'not', 'Not', 'no'}

def preprocess_text_NA_try(text,allowed_postags=["NOUN", "ADJ", "VERB", "ADV","PART"]):
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
        
        # replace abbreviation with words (e.g., "lol" with "laughing out loudly")
        abbreviations = {'lol': 'laughing out loudly'}  # Add your abbreviation dictionary
        words_abbr_removed = [abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word for word in words]

        # Remove stopwords, punctuation, digits, and perform lemmatization 
        spacy_doc = nlp(" ".join(words_abbr_removed))
        transformed_words = [token.lemma_ for token in spacy_doc
                             if token.pos_ in allowed_postags
                             and token.lemma_ not in custom_stopwords
                             and token.lemma_ not in string.punctuation
                             and not token.lemma_.isdigit()]

        # Join the transformed words into a sentence
        transformed_sentence = ' '.join(transformed_words)

        transformed_sentences.append(transformed_sentence)

    # Join the transformed sentences into a single string
    transformed_text = ' '.join(transformed_sentences)

    return transformed_text

# Example usage
text = "never purchasing this phone again. dont buy this phone"
preprocessed_text = preprocess_text_NA_try(text)

print(preprocessed_text)


# In[149]:





# In[150]:


import spacy

nlp = spacy.load("en_core_web_sm")

def check_allowed_postag(text, allowed_postags):
    doc = nlp(text)
    
    for token in doc:
        if token.pos_ in allowed_postags:
            print(f"Word: {token.text}, POS Tag: {token.pos_}, Allowed POS: {allowed_postags}")
        else:
            print(f"Word: {token.text}, POS Tag: {token.pos_}, Not in Allowed POS: {allowed_postags}")

# Example usage
text = "Camera is not good. Battery is not bad. not no never . dont buy this phone .never purchasing this phone again."
allowed_postags = ["NOUN", "ADJ"]

check_allowed_postag(text, allowed_postags)


# In[128]:


import nltk
nltk.download('wordnet')


# In[331]:


my_dict = {'key1': 1, 'key2': 2, 'key3': 3}

# Using keys() method
keys = my_dict.keys()
print(keys)  # Output: dict_keys(['key1', 'key2', 'key3'])

# Converting dictionary to a list
keys_list = list(my_dict)
print(keys_list)  # Output: ['key1', 'key2', 'key3']


# In[333]:


my_dict['key1']


# In[ ]:


my_dict.


# In[364]:


def analyse_data(data):
    columns = ['Review','Aspect','Sentiment']
    #reviews = list(filter (None, data.strip().split('. ')))
    #reviews = [basic_data_preprocessor (review) for review in reviews]
    #reviews = [advanced_data_preprocessor (review) for review in reviews]
    reviews = preprocess_text_NA_SA(data.strip())
    #cv_model = pickle.load(open ("cv model.sav",'rb'))   ---1
    # vector = cv_model.transform(reviews)  ----2 
    

    review_vector = [sent_vec(sent,word_2_vec_model) for sent in reviews]


# In[369]:


analyse_data("hello how are you doing? . i hope you are doing well. Mobile camera is awesome")


# In[368]:


def sent_vec(sent,model_wv):
    sent = word_tokenize(sent)
    #print(sent)
    vector_size = model_wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += model_wv[w]
    wv_res = wv_res/ctr
    return wv_res


# # testing streamlit

# In[170]:


import pickle

with open("Aspect_classifier_model",'rb') as file:
    aspect_model = pickle.load(file)
with open("Sentiment_supportVectorClassifier",'rb' ) as file:
    sent_model  = pickle.load(file)

with open('word_2_vec_model','rb') as file:
    word_2_vec_model  = pickle.load(file)


# In[336]:


data = 'picture clicked'


# In[337]:


reviews = preprocess_text_NA_SA(data)
reviews


# In[338]:


review_vector = [sent_vec(sent,word_2_vec_model) for sent in reviews if len(sent)>=1]


# In[339]:


len(review_vector)


# In[340]:


review_vector


# In[341]:


as_pred = classifier_1.predict(review_vector)
as_pred


# In[242]:


i for i in review_vector if  i not null


# In[239]:


classifier_1.predict(review_vector)


# In[233]:


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


# In[176]:


first = sent_vec('yehllooo lloooo wadda vooo nakko pakko',word_2_vec_model)
first


# In[174]:


len(review_vector)


# In[231]:


classifier_1.predict(review_vector)


# In[257]:


with open("Aspect_classifier_model",'rb') as file:
    aspect_model = pickle.load(file)
aspect_model


# In[258]:


aspect_model.predict(review_vector)


# In[223]:


= df_Review_sentiment['Word2vec']
type(x)


# In[224]:


X = df_Review_sentiment['Word2vec'].to_list()
type(X)


# In[321]:


df_Review_sentiment


# In[274]:


df.to_csv("check_df",index= True )


# In[ ]:


df = df.reset_index()
df =  df.rename(columns={'index': 'Numbering'})
df


# In[322]:


df_merged = df.merge(df_Review_sentiment, how='left', left_on='Numbering', right_on='Numbering')


# In[323]:


df_merged.tail(10)


# In[324]:


df_merged.info()


# In[325]:


df_merged = df_merged[df_merged['Aspect'].notna() & df_merged['Sentiment'].notna()]
df_merged.info()


# In[330]:


df_merged.to_csv("Dataframe_combined_for_exploration.csv",index= True)


# In[327]:


len(df['Product_Name'].unique())


# In[315]:


df['Product_Name'].unique()


# In[328]:


import seaborn as sns
sns.barplot( x='Sentiment_target', y="Aspect", data= df_merged[df_merged['Product_Name']=='SAMSUNG Galaxy A54 5G A Series Cell Phone, Factory Unlocked Android Smartphone, 128GB w/ 6.4” Fluid Display Screen, Hi Res Camera, Long Battery Life, Refined Design, US Version, 2023, Awesome Black'])


# In[329]:


df_merged =  df_merged.rename(columns={'Sentiment_target': 'Sentiment Value'})
df_merged.info()


# In[331]:


get_ipython().system('pip install streamlit')


# In[332]:


get_ipython().system('pip install -U ipykernel')


# In[333]:


get_ipython().system('pip install pyngrok')


# In[ ]:




