from typing_extensions import runtime
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report


class SvmModel:
    def svm():
        cancer = pd.read_csv('data.csv')
        cancer.drop(['Unnamed: 32','id'],axis=1, inplace=True)

        x = cancer.drop(['diagnosis'],axis =1)

        y = cancer ['diagnosis']

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        svc_model= SVC()

        svc_model.fit(x_train,y_train)

        y_predict =svc_model.predict(x_test)


        mini_train =x_train.min()
        rt_train =(x_train - mini_train).max()
        x_train_s =(x_train-mini_train)/rt_train

        mini_test =x_test.min()
        rt_test =(x_test - mini_test).max()
        x_test_s =(x_test-mini_test)/rt_test

        svc_model.fit(x_train_s,y_train)

        y_predict =svc_model.predict(x_test_s)

        # print(classification_report(y_test,y_predict))

        param_grid ={'C':[0.1,.5,.75,.9,1],'gamma':[1,0.1,0.01,0.001]}

        from sklearn.model_selection import GridSearchCV
        grid_s=GridSearchCV(SVC(),param_grid,refit=True,verbose=1)

        grid_s.fit(x_train_s,y_train)

        grid_s.best_params_

        grid_predict=grid_s.predict(x_test_s)

        # return classification_report(y_test,grid_predict,target_names=['Malignant', 'Benign'])
        return classification_report(y_test, grid_predict, target_names=['Malignant', 'Benign'], output_dict=True)