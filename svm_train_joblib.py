
from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.externals import joblib
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

# training_object_collection = []
# training_text_collection = []

# for text in training_collection:
#   gc.disable()
#   training_object_collection.append([text.id, text.period_start_year])
#   training_text_collection.append(text.text_content)
#   gc.enable()

# save_training_object_collection = open("training_object_collection.pkl", "wb")
# joblib.dump(training_object_collection, save_training_object_collection)
# save_training_object_collection.close()

# save_training_text_collection = open("training_text_collection.pkl", "wb")
# joblib.dump(training_text_collection, save_training_text_collection)
# save_training_text_collection.close()

training_object_collection_f = open("training_object_collection.pkl", "rb")
training_object_collection = joblib.load(training_object_collection_f)
training_object_collection_f.close()

training_text_collection_f = open("training_text_collection.pkl", "rb")
training_text_collection = joblib.load(training_text_collection_f)
training_text_collection_f.close()

testing_collection = Text.query.filter_by(data_set = "test").all()

testing_object_collection = []
testing_text_collection = []

for text in testing_collection:
  gc.disable()
  testing_object_collection.append([text.id, text.period_start_year, text.gutenberg_id, text.author_birth_year])
  testing_text_collection.append(text.text_content)
  gc.enable()

save_testing_object_collection = open("testing_object_collection.pkl", "wb")
joblib.dump(testing_object_collection, save_testing_object_collection)
save_testing_object_collection.close()

save_testing_text_collection = open("testing_text_collection.pkl", "wb")
joblib.dump(testing_text_collection, save_testing_text_collection)
save_testing_text_collection.close()

# testing_object_collection_f = open("testing_object_collection.pickle", "rb")
# testing_object_collection = pickle.load(testing_object_collection_f)
# testing_object_collection_f.close()

# testing_text_collection_f = open("testing_text_collection.pickle", "rb")
# testing_text_collection = pickle.load(testing_text_collection_f)
# testing_text_collection_f.close()

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = "english",   \
                             max_features = 2500, \
                             min_df = 5, \
                             max_df = 0.5)

print("Vectorizing data...")


# vect_f = open("nb_vect.pickle", "rb")
# vectorizer = pickle.load(vect_f)
# vect_f.close()


train_data_features = vectorizer.fit_transform(training_text_collection)
train_data_features = train_data_features.toarray()

save_vect = open("svm_vect.pkl", "wb")
joblib.dump(vectorizer, save_vect)
save_vect.close()

tfidf_transformer = TfidfTransformer()


# tfidf_f = open("nb_tfidf.pickle", "rb")
# tfidf_transformer = pickle.load(tfidf_f)
# tfidf_f.close()

train_tfidf = tfidf_transformer.fit_transform(train_data_features)

save_tfidf = open("svm_tfidf.pkl", "wb")
joblib.dump(tfidf_transformer, save_tfidf)
save_tfidf.close()

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

# clf_f = open("svm_clf.pickle", "rb")
# clf = pickle.load(clf_f)
# clf_f.close()

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42).fit(train_tfidf, training_targets)

save_clf = open("svm_clf.pkl", "wb")
joblib.dump(clf, save_clf)
save_clf.close()

testing_data_features = vectorizer.transform(testing_text_collection)
new_tfidf = tfidf_transformer.transform(testing_data_features)

midpoint = datetime.datetime.now()
before_prediction = midpoint - start_of_program
print "Time taken until prediction begun:", before_prediction

print "Making prediction..."

predicted = clf.predict(new_tfidf)
mean = np.mean(predicted == testing_targets)
print(mean)
# print(type(predicted))

print(metrics.confusion_matrix(testing_targets, predicted))
print(metrics.classification_report(testing_targets, predicted))

end_of_program = datetime.datetime.now()
print "End of program:", end_of_program

total_time = end_of_program - start_of_program
print "Runtime:", datetime.datetime.now() - start_of_program
