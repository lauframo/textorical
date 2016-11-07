from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import datetime
import gc


start_of_program = datetime.datetime.now()
print "Start of program:", start_of_program

# def get_training_set(year):
#   return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").all()

# def get_testing_set(year):
#   return Text.query.filter_by(period_start_year = year).filter_by(data_set = "test").all()
print "Grabbing data..."
training_collection = Text.query.filter_by(data_set = "train").all()
testing_collection = Text.query.filter_by(data_set = "test").all()

object_collection = []
text_collection = []

for text in training_collection:
  gc.disable()
  object_collection.append([text.id, text.period_start_year, text.text_content])
  text_collection.append(text.text_content)
  gc.enable()

testing_object_collection = []
testing_text_collection = []

for text in testing_collection:
  gc.disable()
  testing_object_collection.append([text.id, text.period_start_year, text.text_content, text.gutenberg_id, text.author_birth_year])
  testing_text_collection.append(text.text_content)
  gc.enable()


# df = DataFrame(object_collection)
# df.columns = ['id', 'period_start_year', 'text_content']

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing data...")

train_data_features = vectorizer.fit_transform(text_collection)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()


tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_data_features)


training_targets = []
for text in object_collection:
  gc.disable()
  training_targets.append(text[1])
  gc.enable()

testing_targets = []
for text in testing_object_collection:
  gc.disable()
  testing_targets.append(text[1])
  gc.enable()


premidpoint = datetime.datetime.now()
before_fit = premidpoint - start_of_program
print "Time taken until fit is begun:", before_fit

clf = MultinomialNB().fit(train_tfidf, training_targets)

testing_data_features = vectorizer.transform(testing_text_collection)
new_tfidf = tfidf_transformer.transform(testing_data_features)

midpoint = datetime.datetime.now()
before_prediction = midpoint - start_of_program
print "Time taken until prediction begun:", before_prediction

print "Making prediction..."

predicted = clf.predict(new_tfidf)
mean = np.mean(predicted == testing_targets)
print(mean)
print(type(predicted))

end_of_program = datetime.datetime.now()
print "End of program:", end_of_program

total_time = end_of_program - start_of_program
print "Runtime:", total_time
