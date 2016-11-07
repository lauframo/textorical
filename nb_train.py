from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer

def get_training_set(year):
  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").limit(20).all()

training_collection = get_training_set(1500) + get_training_set(1600) + get_training_set(1700) + get_training_set(1800) +get_training_set(1900)

object_collection = []
text_collection = []

for text in training_collection:
  object_collection.append([text.id, text.period_start_year, text.text_content])
  text_collection.append(text.text_content)

df = DataFrame(object_collection)
df.columns = ['id', 'period_start_year', 'text_content']

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing data...")
train_data_features = vectorizer.fit_transform(text_collection)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
# print(vocab)

# tf_transformer = TfidfTransformer(use_idf=False).fit(train_data_features)
# train_tf = tf_transformer.transform(train_data_features)
# print(train_tf)

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_data_features)
print(train_tfidf.shape)
