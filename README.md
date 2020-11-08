import pandas as pd
import numpy as np
data_train = pd.read_csv(r"C:\Users\user\Downloads\train_data_cleaning.csv")
data_test = pd.read_csv(r"C:\Users\user\Downloads\test_data_cleaning.csv")
data_test.head()
data_train.head()	
data_train.shape
data_test.shape
data_train.drop(["keyword","location"],axis=1, inplace=True)
data_test.drop(["keyword","location"],axis=1, inplace=True)
data_train.head()
data_test.head()
data_train['text_lower']= data_train['text'].apply(lambda x: x.lower())
data_test['text_lower']= data_test['text'].apply(lambda x: x.lower())
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
st_words = stopwords.words('english')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
nltk.download('wordnet')
lemmatizerwrd= WordNetLemmatizer()
data_train['text_lower']= data_train['text_lower'].apply(lambda x: " ".join([lemmatizerwrd.lemmatize(word) for word in x.split(" ")]))
data_test['text_lower']= data_test['text_lower'].apply(lambda x: " ".join([lemmatizerwrd.lemmatize(word) for word in x.split(" ")]))
data_train['text_lower_new']= data_train['text_lower'].apply(lambda x: " ".join([word for word in x.split(" ") if word not in st_words]))
data_test['text_lower_new']= data_test['text_lower'].apply(lambda x: " ".join([word for word in x.split(" ") if word not in st_words]))
import re
data_train["text_lower_new"] = data_train["text_lower_new"].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
data_test["text_lower_new"] = data_test["text_lower_new"].apply(lambda x: re.sub('[^a-zA-Z]+', ' ', x))
from sklearn.feature_extraction.text import CountVectorizer
CV1 = CountVectorizer()
CV2 = CountVectorizer()
CV1.fit(data_train['text_lower_new'])
CV2.fit(data_test['text_lower_new'])
X1 = CV1.transform(data_train['text_lower_new'])
X2 = CV1.transform(data_test['text_lower_new'])	
transformedX1 = pd.DataFrame(X1.toarray(), columns=CV1.get_feature_names())
transformedX2 = pd.DataFrame(X2.toarray(), columns=CV1.get_feature_names())
transformedX1.head()
transformedX2.head()
trainx=transformedX1
testx=transformedX2
trainy=data_train.iloc[:,2]
trainy.head()
testy=data_test.iloc[:,2]
testy.head()
from sklearn.naive_bayes import MultinomialNB 
model = MultinomialNB()
model.fit(trainx,trainy)
pred_y= model.predict(testx)
from sklearn.metrics import classification_report
print(classification_report(pred_y, testy))
