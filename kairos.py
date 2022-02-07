import graphlib
import streamlit as st
import pandas as pd

#Gensim modules
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Initialize df container
global data2
global teacher_dropdown
data2 = pd.DataFrame(columns=['Course','Teacher','Text', 'Classification'])

if "load_state" not in st.session_state:
	st.session_state.load_state = False

#POS Tagging
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
wn.ensure_loaded()
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_stop_pos(text):
	tags = pos_tag(word_tokenize(text))
	newlist = []
	for word, tag in tags:
		if word.lower() not in set(stopwords.words('english')):
			newlist.append(tuple([word, pos_dict.get(tag[0])]))
	return newlist

#Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
	lemma_rew = " "
	for word, pos in pos_data:
		if not pos:
			lemma = word
			lemma_rew = lemma_rew + " " + lemma
		else:
			lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
			lemma_rew = lemma_rew + " " + lemma
	return lemma_rew

#Sentiment analysis using TextBlob
from textblob import TextBlob
# function to calculate subjectivity
def getSubjectivity(review):
	return TextBlob(review).sentiment.subjectivity
# function to calculate polarity
def getPolarity(review):
	return TextBlob(review).sentiment.polarity

# function to analyze the reviews
def analysis(score):
	if score < 0:
		return 'Negative'
	elif score == 0:
		return 'Neutral'
	else:
		return 'Positive'

#Plot
def bar_graph_course(df, value):
	df_course = df[df['Course'] ==  value]['Classification'].value_counts()
	st.bar_chart(df_course)

def bar_graph_teacher(df, value):
	df_teacher = df[df['Teacher'] == value]['Classification'].value_counts()
	st.bar_chart(df_teacher)

#LSA functions
#prepare corpus
def prepare_corpus(classified_text):
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(classified_text)
    dictionary = corpora.Dictionary(classified_text)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in classified_text]
    # generate LDA model
    return dictionary,doc_term_matrix

#create lsa model
def create_gensim_lsa_model(classified_text,number_of_topics,words):
    dictionary,doc_term_matrix=prepare_corpus(classified_text)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    return lsamodel

def preprocess_data(doc_set):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts

@st.cache
def load_data():
	#Initialize df container
	data2 = data[['Text', 'Classification','Teacher','Course']]
	return data2

header = st.container()
upload = st.container()
classify = st.container()
teacher = st.container()
course = st.container()


with header: 
	st.title('SAT Comment Classifier')
	st.text('Presented to you by Kairos')


with upload:
	st.title('Upload')
	
	uploaded_file = st.file_uploader("Upload", type='csv')
	if uploaded_file is not None:
		data = pd.read_csv(uploaded_file)
		data.columns=['Course','Teacher','Text']
		st.dataframe(data)
	else:
		st.text('No file uploaded')

with classify:
	st.title('Classify')
	if st.button('Classify Data') or st.session_state.load_state:
		st.session_state.load_state = True
		data['POS tagged'] = data['Text'].apply(token_stop_pos)
		data['Lemma'] = data['POS tagged'].apply(lemmatize)
		data['Text_Blob Polarity'] = data['Lemma'].apply(getPolarity) 
		data['Classification'] = data['Text_Blob Polarity'].apply(analysis)
		data2 = load_data()
		data2 = data2.sort_values('Classification')
		st.dataframe(data2[['Text','Classification']])
		
		#Insert Topics
		#LSA
		#grouping positive classifications
		positive_df = data2.loc[data2['Classification'] == 'Positive']
		#grouping neutral classifications
		neutral_df = data2.loc[data2['Classification'] == 'Neutral']
		#grouping negative classifications
		negative_df = data2.loc[data2['Classification'] == 'Negative']
		#creating the LSA models for the different classifications
		number_of_topics = 3
		words = 30
		clean_positive = preprocess_data(positive_df['Text'])
		clean_neutral = preprocess_data(neutral_df['Text'])
		clean_negative = preprocess_data(negative_df['Text'])

		#positive LSA
		st.title('Positive Topics')
		positive_model=create_gensim_lsa_model(clean_positive,number_of_topics,words)
		for index, topic in positive_model.show_topics(num_topics=number_of_topics, num_words=words, formatted = False):
    			st.text('Topic: {} \nWords: {}'.format(index+1, '|'.join([w[0] for w in topic])))
		
		#negative LSA
		st.title('Negative Topics')
		negative_model =create_gensim_lsa_model(clean_negative,number_of_topics,words)
		for index, topic in negative_model.show_topics(num_topics=number_of_topics, num_words=words, formatted = False):
    			st.text('Topic: {} \nWords: {}'.format(index+1, '|'.join([w[0] for w in topic])))

		#neutral LSA
		st.title('Neutral Topics')
		neutral_model=create_gensim_lsa_model(clean_neutral,number_of_topics,words)
		for index, topic in neutral_model.show_topics(num_topics=number_of_topics, num_words=words, formatted = False):
    			st.text('Topic: {} \nWords: {}'.format(index+1, '|'.join([w[0] for w in topic])))

with teacher:
	teacher_dropdown = st.selectbox('Teachers', data2['Teacher'].unique())
	bar_graph_teacher(data2, teacher_dropdown)

with course:
	courses_dropdown = st.selectbox('Courses', data2['Course'].unique())
	bar_graph_course(data2, courses_dropdown)
	#&& data2['Teacher'] == 'teacher_dropdown'
