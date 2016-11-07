from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def get_training_set(year):
  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").limit(20).all()

def get_testing_set(year):
  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "test").limit(20).all()

training_collection = get_training_set(1500) + get_training_set(1600) + get_training_set(1700) + get_training_set(1800) +get_training_set(1900)
testing_collection = get_testing_set(1500) + get_testing_set(1600) + get_testing_set(1700) + get_testing_set(1800) + get_testing_set(1900)

object_collection = []
text_collection = []

for text in training_collection:
  object_collection.append([text.id, text.period_start_year, text.text_content])
  text_collection.append(text.text_content)

testing_object_collection = []
testing_text_collection = []

for text in testing_collection:
  testing_object_collection.append([text.id, text.period_start_year, text.text_content])
  testing_text_collection.append(text.text_content)

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


training_targets = []
for text in object_collection:
  training_targets.append(text[1])

testing_targets = []
for text in testing_object_collection:
  testing_targets.append(text[1])

clf = MultinomialNB().fit(train_tfidf, training_targets)

# testing_text = Text.query.filter_by(data_set='test').filter_by(period_start_year=1800).first()
# testing_text = testing_text.text_content

testing_data_features = vectorizer.transform(testing_text_collection)
new_tfidf = tfidf_transformer.transform(testing_data_features)

predicted = clf.predict(new_tfidf)
mean = np.mean(predicted == testing_targets)
print(mean)
