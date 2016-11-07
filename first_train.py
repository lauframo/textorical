from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame


def get_training_set(year):
  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "train").limit(20).all()

training_collection = get_training_set(1500) + get_training_set(1600) + get_training_set(1700) + get_training_set(1800) +get_training_set(1900)

train_object_collection = []
train_text_collection = []

for text in training_collection:
  train_object_collection.append([text.id, text.period_start_year, text.text_content])
  train_text_collection.append(text.text_content)

train = DataFrame(train_object_collection)
train.columns = ['id', 'period_start_year', 'text_content']

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing training data...")
train_data_features = vectorizer.fit_transform(train_text_collection)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

print("Training the random forest...")
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["period_start_year"])

print forest.n_features_
print forest.classes_
print forest.feature_importances_

############################################
#Test
############################################

def get_validation_set(year):
 return Text.query.filter_by(period_start_year = year).filter_by(data_set = "validation").limit(20).all()

validation_collection = get_validation_set(1500) + get_validation_set(1600) + get_validation_set(1700) + get_validation_set(1800) +get_validation_set(1900)

test_object_collection = []
test_text_collection = []

for text in validation_collection:
  test_object_collection.append([text.id, text.gutenberg_id, text.title, text.author_birth_year, text.period_start_year, text.text_content])
  test_text_collection.append(text.text_content)

test = DataFrame(test_object_collection)
test.columns = ['id', 'gutenberg_id', 'title', 'author_birth_year', 'period_start_year', 'text_content']

print("Vectorizing test data...")
test_data_features = vectorizer.transform(test_text_collection)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = DataFrame(data={"id":test["id"], "gutenberg_id":test["gutenberg_id"], "author_birth_year":test["author_birth_year"], "period_start_year":result, "accurate":(test["period_start_year"]==result) })

accuracy = forest.score(test_data_features, test["period_start_year"], sample_weight=None)
print "Accuracy: ", accuracy

output.to_csv("Bag_of_Words_modelv02.csv", index=False, quoting=3 )
