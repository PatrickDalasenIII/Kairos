import graphlib
import streamlit as st
import pandas as pd

#Initialize df container
global data2
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
nltk.download('punkt')
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

with teacher:
	teacher_dropdown = st.selectbox('Teachers', data2['Teacher'].unique())
	bar_graph_teacher(data2, teacher_dropdown)

with course:
	courses_dropdown = st.selectbox('Courses', data2['Course'].unique())
	bar_graph_course(data2, courses_dropdown)
