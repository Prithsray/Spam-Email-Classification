import matplotlib.pyplot as plt
import nltk 
import numpy as np
import pandas as pd 
import seaborn as sns 
import sklearn.naive_bayes
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics


data = pd.read_csv(r"D:/COLLEGEMATERIALS/Project Me/Spam_mail_detection/Spam email dataset/spam.csv",encoding="ISO-8859-1")
data['v1'] = np.where(data['v1']=='spam',1, 0)
data.head(10)

Text_list =data['v2'].tolist()
#Spam_ham_list = data['v1'].tolist()

#print(Spam_ham_list)

X_train, X_test, Y_train, Y_test = train_test_split(Text_list, data['v1'], random_state=0)

vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
X_train_vectorized.toarray().shape

#print(data.head(20))

#print(X_train_vectorized)
#print(Y_train_vectorized[0])

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)

predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == Y_test) / len(predictions), '%')
print(metrics.classification_report(list(Y_test), predictions))



