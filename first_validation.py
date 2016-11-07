from app import db
from app import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pandas import DataFrame
import first_train

def get_validation_set(year):
 return Text.query.filter_by(period_start_year = year).filter_by(data_set = "validation").limit(20).all()

validation_collection = get_validation_set(1500) + get_validation_set(1600) + get_validation_set(1700) + get_validation_set(1800) +get_validation_set(1900)

object_collection = []
text_collection = []

for text in validation_collection:
  object_collection.append([text.id, text.period_start_year, text.text_content])
  text_collection.append(text.text_content)

test = DataFrame(object_collection)
test.columns = ['id', 'title', 'author_birth_year','period_start_year', 'text_content']

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print("Vectorizing data...")
test_data_features = vectorizer.transform(text_collection)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pandas.DataFrame(data={"id":test["id"], "title":test["title"],"period_start_year":result })

output.to_csv("Bag_of_Words_modelv01.csv", index=False, quoting=3 )

