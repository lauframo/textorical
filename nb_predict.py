import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

vect_f = open("nb_vect.pickle", "rb")
nb_vectorizer = pickle.load(vect_f)
vect_f.close()

tfidf_f = open("nb_tfidf.pickle", "rb")
nb_tfidf_transformer = pickle.load(tfidf_f)
tfidf_f.close()

clf_f = open("nb_clf.pickle", "rb")
nb_clf = pickle.load(clf_f)
clf_f.close()

def nb_predict(text_to_check):
  vectorized_text = nb_vectorizer.transform([text_to_check])
  text_features = nb_tfidf_transformer.transform(vectorized_text)
  return  nb_clf.predict(text_features)

print nb_predict("dummy")
