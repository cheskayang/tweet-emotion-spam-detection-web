import pickle
import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from feature_extraction import processData


dest = os.path.join('pickled_classifier', 'pkl_objects')


clf = pickle.load(open(os.path.join(dest, 'classifier.pkl'), 'rb'))
vect = pickle.load(open(os.path.join(dest, 'vect.pkl'), 'rb'))


label = {0:'non-spam', 1:'spam'}
example = ['CHECK IT OUT A1:S4000 #Anger Inside Out Small #Figure #sales #InsideOut #ebay #shopping']
processedExample = processData(example)
X = vect.transform(processedExample)
print(clf.predict(X))
