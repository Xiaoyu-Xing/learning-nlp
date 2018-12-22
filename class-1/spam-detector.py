from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd 
import numpy as np 

data = pd.read_csv('../dataset/spam/spambase.data').values
np.random.shuffle(data)
X = data[:, :48]
Y = data[:, -1]
X_train = X[:-200,]
Y_train = Y[:-200,]
X_test = X[-200:,]
Y_test = Y[-200:,]

def train(model):
	model = model()
	model.fit(X_train, Y_train)
	print("Classification rate: ", model.score(X_test, Y_test))

train(MultinomialNB)
train(AdaBoostClassifier)
