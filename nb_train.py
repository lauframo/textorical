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
import cPickle as pickle


start_of_program = datetime.datetime.now()
print "Start of program:", start_of_program

# def get_training_set(year):
#   return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").all()

# def get_testing_set(year):
#   return Text.query.filter_by(period_start_year = year).filter_by(data_set = "test").all()
print "Grabbing data..."
# training_collection = Text.query.filter_by(data_set = "train").all()
# testing_collection = Text.query.filter_by(data_set = "test").all()

# training_object_collection = []
# training_text_collection = []

# for text in training_collection:
#   gc.disable()
#   training_object_collection.append([text.id, text.period_start_year])
#   training_text_collection.append(text.text_content)
#   gc.enable()

# save_training_object_collection = open("training_object_collection.pickle", "wb")
# pickle.dump(training_object_collection, save_training_object_collection)
# save_training_object_collection.close()

# save_training_text_collection = open("training_text_collection.pickle", "wb")
# pickle.dump(training_text_collection, save_training_text_collection)
# save_training_text_collection.close()

training_object_collection_f = open("training_object_collection.pickle", "rb")
training_object_collection = pickle.load(training_object_collection_f)
training_object_collection_f.close()

training_text_collection_f = open("training_text_collection.pickle", "rb")
training_text_collection = pickle.load(training_text_collection_f)
training_text_collection_f.close()

# testing_object_collection = []
# testing_text_collection = []

# for text in testing_collection:
#   gc.disable()
#   testing_object_collection.append([text.id, text.period_start_year, text.gutenberg_id, text.author_birth_year])
#   testing_text_collection.append(text.text_content)
#   gc.enable()

# save_testing_object_collection = open("testing_object_collection.pickle", "wb")
# pickle.dump(testing_object_collection, save_testing_object_collection)
# save_testing_object_collection.close()

# save_testing_text_collection = open("testing_text_collection.pickle", "wb")
# pickle.dump(testing_text_collection, save_testing_text_collection)
# save_testing_text_collection.close()

testing_object_collection_f = open("testing_object_collection.pickle", "rb")
testing_object_collection = pickle.load(testing_object_collection_f)
testing_object_collection_f.close()

testing_text_collection_f = open("testing_text_collection.pickle", "rb")
testing_text_collection = pickle.load(testing_text_collection_f)
testing_text_collection_f.close()


vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing data...")

train_data_features = vectorizer.fit_transform(training_text_collection)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()


tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_data_features)


training_targets = []
for text in training_object_collection:
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
