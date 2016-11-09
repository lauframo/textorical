import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

vect_f = open("svm_vect.pickle", "rb")
vectorizer = pickle.load(vect_f)
vect_f.close()

tfidf_f = open("svm_tfidf.pickle", "rb")
tfidf_transformer = pickle.load(tfidf_f)
tfidf_f.close()

clf_f = open("svm_clf.pickle", "rb")
clf = pickle.load(clf_f)
clf_f.close()

def svm_predict(text_to_check):
  vectorized_text = vectorizer.transform([text_to_check])
  text_features = tfidf_transformer.transform(vectorized_text)
  return  clf.predict(text_features)

print svm_predict("dummy")
