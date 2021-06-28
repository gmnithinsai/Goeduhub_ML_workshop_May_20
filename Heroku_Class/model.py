
import pandas as pd
import numpy as np
import pickle


df=pd.read_csv(r'E:\Goeduhub_ML_Program_May_20\data\hiring.csv')

df['experience'].fillna(0,inplace=True)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(),inplace=True)

def string_to_num(word):
    dict={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,0:0}
    return dict[word]
df['experience']=df['experience'].apply(lambda x : string_to_num(x))

x=df.iloc[:,:3]
y=df.iloc[:,-1]

#from sklearn.model_selection import train_test_split
#xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.26,random_state=5)

from sklearn.linear_model import LinearRegression
mymodel=LinearRegression()

mymodel.fit(x,y)


pickle.dump(mymodel,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([2,9,6])))