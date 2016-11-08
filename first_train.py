from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from pandas import DataFrame
import random
import datetime
import gc

start_of_program = datetime.datetime.now()
print "Start of program:", start_of_program

# def get_training_set(year):
#   return db.engine.execute("SELECT * FROM texts where data_set = 'train' limit 200" )


training_collection = db.engine.execute("SELECT id, period_start_year, text_content FROM texts where data_set = 'train'" )
# training_collection = Text.query.filter_by(data_set = "train").limit(2000).all()

# training_collection = get_training_set(1500) + get_training_set(1600) + get_training_set(1700) + get_training_set(1800) +get_training_set(1900)


train_object_collection = []
train_text_collection = []

gc.disable()
for text in training_collection:
  train_object_collection.append([text[0], text[1], text[2]])
  train_text_collection.append(text[2])
gc.enable()


train = DataFrame(train_object_collection)
train.columns = ['id', 'period_start_year', 'text_content']

# middle_of_program = datetime.datetime.now()
# print "Before appending:", middle_of_program

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing training data...")
train_data_features = vectorizer.fit_transform(train_text_collection)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_data_features)

print("Training the random forest...")
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_tfidf, train["period_start_year"])

# print forest.n_features_
# print forest.classes_
# print forest.feature_importances_

# ############################################
# #Test
# ############################################

# def get_validation_set(year):
#  return Text.query.filter_by(period_start_year = year).filter_by(data_set = "validation").limit(20).all()

validation_collection = db.engine.execute("SELECT id, gutenberg_id, author_birth_year, period_start_year, text_content FROM texts where data_set = 'validation'" )

# print "Validation Collection:", validation_collection

validation_object_collection = []
validation_text_collection = []

gc.disable()
for text in validation_collection:
  validation_object_collection.append([text[0], text[1], text[2], text[3], text[4]])
  validation_text_collection.append(text[4])
gc.enable()

# random.shuffle(test_object_collection)

validation = DataFrame(validation_object_collection)
validation.columns = ['id', 'gutenberg_id', 'author_birth_year', 'period_start_year', 'text_content']

print("Vectorizing validation data...")
validation_data_features = vectorizer.transform(validation_text_collection)
validation_data_features = validation_data_features.toarray()

new_tfidf = tfidf_transformer.transform(validation_data_features)


result = forest.predict(new_tfidf)

output = DataFrame(data={"id":validation["id"], "gutenberg_id":validation["gutenberg_id"], "author_birth_year":validation["author_birth_year"], "period_start_year":result, "accurate":(validation["period_start_year"]==result) })

accuracy = forest.score(validation_data_features, validation["period_start_year"], sample_weight=None)
print "Accuracy: ", accuracy

output.to_csv("RandomForest_Bag_of_Words_modelv05.csv", index=False, quoting=3 )

end_of_program = datetime.datetime.now()
# print "End of program:", end_of_program

# part_time = end_of_program - middle_of_program
# print "Append Runtime:", part_time

total_time = end_of_program - start_of_program
print "Runtime:", total_time

