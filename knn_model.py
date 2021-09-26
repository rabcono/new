import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class KnnModel:
    def knn():
        cancer = pd.read_csv('data.csv')

        cancer.isnull().sum()

        cancer.drop(['Unnamed: 32','id'],axis=1, inplace=True)

        wod=cancer.drop('diagnosis',axis=1)

        s_scaled=StandardScaler()

        x=pd.DataFrame(s_scaled.fit_transform(wod),columns=wod.columns)

        y=cancer['diagnosis'].map({'M':0,'B':1}).astype(int)

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

        error_rate=[]

        for i in range(1,45):
            knn=KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train,y_train)
            pred_i=knn.predict(x_test)
            error_rate.append(np.mean(pred_i) != y_test)
            
        KN=np.arange(1,35)
        train_acc=np.empty(len(KN))
        test_acc=np.empty(len(KN))

        for i,k in enumerate(KN):
            knn=KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train,y_train)
            train_acc[i]=knn.score(x_train,y_train)
            test_acc[i]=knn.score(x_test,y_test)

        dex=np.where(test_acc==max(test_acc))
        # f=[dex]

        model=KNeighborsClassifier(n_neighbors=9)
        model.fit(x_train,y_train)

        y_pred=model.predict(x_test)

        # return classification_report(y_test,y_pred,target_names=['Malignant', 'Benign'])
        return classification_report(y_test, y_pred, target_names=['Malignant', 'Benign'],output_dict=True,)