import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class LogisticModel:
    def logistic():
        cancer = pd.read_csv('data.csv')

        cancer.drop(['Unnamed: 32','id'],axis=1, inplace=True)

        x = cancer.drop(['diagnosis'],axis =1)
        y = cancer ['diagnosis']

        y=cancer['diagnosis'].map({'M':0,'B':1}).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        s_scaled=StandardScaler()
        x_train = s_scaled.fit_transform(x_train)
        x_test = s_scaled.transform(x_test)

        classifier = LogisticRegression().fit(x_train,y_train)
        y_pred = classifier.predict(x_test)

        # return classification_report(y_test, y_pred,target_names=['Malignant', 'Benign'])
        return classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'], output_dict=True)

