import graphlib
import streamlit as st
import pandas as pd
from PIL import Image
image = Image.open('SLU_logo.jpg')
image2 = Image.open('SAMCISv2_logo.jpg')

#Gensim modules
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Initialize df container
global data2
global data5
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
def bar_graph_course(df, value, value2, is_restricted):
	if is_restricted:
		df_course = df[(df['Course'] ==  value) & (df['Teacher'] == value2)]['Classification'].value_counts()
	else:
		df_course = df[df['Course'] ==  value]['Classification'].value_counts()

	st.bar_chart(df_course)

def bar_graph_teacher(df, value, value2, is_restricted):
	if is_restricted:
		df_teacher = df[(df['Teacher'] == value) & (df['Course'] == value2)]['Classification'].value_counts()
	else:
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

def convert_df(df):
   return df.to_csv().encode('utf-8')

header = st.container()
upload = st.container()
classify = st.container()
graph_options = st.container()


with header: 
	col1, col2, col3 = st.columns([1,4.8,1])
	with col1:
		st.image(image, width=100)
	with col2:
		st.write('School of Accountancy, Management, Computer and Information Studies')
	with col3:
		st.image(image2, width=160)		
	st.markdown("<h1 style='text-align: center;'>SET Comment Classifier</h1>", unsafe_allow_html=True)
	st.markdown("<h5 style='text-align: center;'>Presented to you by Kairos</h5>", unsafe_allow_html=True)

#footer:
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by AWANGAN | BANDA-AY | CARILLO | DALASEN | GARCIA | IPORAC | PRESILLAS</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

with upload:
	st.title('Upload')
	
	uploaded_file = st.file_uploader(" ", type='csv')
	if uploaded_file is not None:
		data = pd.read_csv(uploaded_file)
		data.columns=['Course','Teacher','Text']
		sort_value = st.selectbox('Sort by', ('No Option Selected','Course', 'Teacher'))
		if sort_value == 'Course':
			st.dataframe(data.sort_values('Course'))
		elif sort_value == 'Teacher':
			st.dataframe(data.sort_values('Teacher'))
		else:
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
		data2 = data2.sort_values('Classification', ascending=False)
		
		sort_value2 = st.selectbox('Filter by',('No Option Selected','Positive','Negative','Neutral'))
		
		if sort_value2 == 'Positive': 
			data5 = data2.loc[data2['Classification']=='Positive'][['Text','Classification']]
		elif sort_value2 == 'Negative':
			data5 = data2.loc[data2['Classification']=='Negative'][['Text','Classification']]
		elif sort_value2 == 'Neutral':
			data5 = data2.loc[data2['Classification']=='Neutral'][['Text','Classification']]
		else:
			data5 = data2[['Text','Classification']]
		st.dataframe(data5)
		csv = convert_df(data5)
		st.download_button(
   			"Press to Download",
   			csv,
   			"file.csv",
   			"text/csv",
   			key='download-csv'
		)
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

def get_teacher_options(list_options):
	teachers_opt  = []
	for comment in data2.itertuples():
		if comment.Course in list_options:
			teachers_opt.append(comment.Teacher)
	data4 = pd.DataFrame(teachers_opt, columns=['Teacher'])
	return data4['Teacher'].unique()

def get_selected_teacher(is_restricted, cd_options):
	with st.container():
		if is_restricted:
			teacher_dropdown = st.selectbox('Teachers', get_teacher_options(cd_options))
		else:
			teacher_dropdown = st.selectbox('Teachers', data2['Teacher'].unique())
	return teacher_dropdown

def get_course_options(list_options):
	courses_opt  = []
	for comment in data2.itertuples():
		if comment.Teacher in list_options:
			courses_opt.append(comment.Course)
	data3 = pd.DataFrame(courses_opt, columns=['Course'])
	return data3['Course'].unique()

def get_selected_course(is_restricted, td_options):
	with st.container():
		if is_restricted:
			courses_dropdown = st.selectbox('Courses', get_course_options(td_options))
		else:
			courses_dropdown = st.selectbox('Courses', data2['Course'].unique())
	return courses_dropdown

def create_graph(graph_dropdown):
	if graph_dropdown == 'All Evaluations':
		teacher = get_selected_teacher(False, [])
		course = get_selected_course(False, [])
		st.title(f"Classifications of {teacher}")
		bar_graph_teacher(data2, teacher, course, False)
		st.title(f"Classifications of {course}")
		bar_graph_course(data2, course, teacher, False)

	elif graph_dropdown == 'Teachers Evaluation':
		teacher2 = get_selected_teacher(False, [])
		course2 = get_selected_course(True, teacher2)
		st.title(f"Classifications of {teacher2}")
		bar_graph_teacher(data2, teacher2, course2, False)
		st.title(f"Classifications of {course2}")
		bar_graph_course(data2, course2, teacher2, True)

	elif graph_dropdown == 'Courses Evaluation':
		course3 = get_selected_course(False, [])
		teacher3 = get_selected_teacher(True, course3)
		st.title(f"Classifications of {course3}")
		bar_graph_course(data2, course3, teacher3, False)
		st.title(f"Classifications of {teacher3}")
		bar_graph_teacher(data2, teacher3, course3, True)
		
	else:
		print("Something happened")

with graph_options:
	graph_dropdown = st.selectbox('Select Evaluation', ['All Evaluations', 'Teachers Evaluation', 'Courses Evaluation'])
	create_graph(graph_dropdown)
