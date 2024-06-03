import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('survey_lung_cancer.csv')
X = dataset.drop(['AGE','GENDER','SHORTNESS OF BREATH','SMOKING','LUNG_CANCER'],axis=1)
y = dataset['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)
KNN = KNeighborsClassifier()
KNN.fit(X, y)
predictions = KNN.predict(X_test)
val3 = (accuracy_score(y_test, predictions)*100)
print("*Accuracy score for KNN: ", val3, "\n")