import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
data=load_breast_cancer()
X=data['data']
Y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.3,random_state=11)

clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
# print(clf.score(X_test,y_test))
# x_new=np.array(random.sample(range(0,50),30))
# print(data['target_names'][clf.predict[(x_new)[0]]])
# x_new=np.array(random.sample(range(0,100),30))
# print(data['target_names'][clf.predict([x_new])])
column_data=np.concatenate([data['data'],data['target'][:,None]],axis=1)
column_names=np.concatenate([data['feature_names'],["Class"]])
df=pd.DataFrame(column_data,columns=column_names)
sns.heatmap(df.corr(),cmap="coolwarm",annot=True,annot_kws={"fontsize":8})
plt.tight_layout()
plt.show()