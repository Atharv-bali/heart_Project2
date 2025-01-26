import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
db = pd.read_csv('sample.csv')
# print(db.head())      Shows the info of top 5 dataset
# print(db.tail())      Shows the info of last 5 dataset
# print(db.describe(include="all").transpose())     Shows all the rows and columns, but if we remove (include = "all") it will only 
# show numeric values.
# print(db.isnull())        Shows that wheather database contains any null values or not (if value is null it will return true otherwise false)
# print(db.isnull().sum())      Shows that how many datavalues have null vlaues. Had dataset had any null values then use db.dropnull()and remove it
db.drop(db[db['RestingBP']==0].index, inplace=True) 
#If we remove (inplace=True) then db will not store the same we have to use some other variable to hold it 
db.drop(db[db['Cholesterol']==0].index, inplace=True)
# print(db.describe(include="all").transpose())
# print(db.groupby(['FastingBS','HeartDisease'])['HeartDisease'].count())  Used to see how many person having fasting Blood Pressure had
#  heart disease and how many dont have. Likewise you can check for gender as well.
db.drop_duplicates(inplace=True)#used to remove duplicates
# print(db.duplicated().sum())  Used to see wheather duplicates are still present or not.
db['Sex'].unique() # .unique tells us that what different gender do we have (if we remove .unique then we will have all male
#  and female present instead of habving just one time)
db['Sex'] = db['Sex'].replace({'M':0,'F':1}) #.replace replaces the male by 0 and female by 1
db['ChestPainType'].unique()
db['ChestPainType'] = db['ChestPainType'].replace({'ATA':0,'NAP':1,'ASY':2,'TA':3})
# print(db['ChestPainType'].value_counts())     value_count tells that number of heart disease for a particular type pf chest pain
db['RestingECG'].unique()
db['RestingECG'] = db['RestingECG'].replace({'Normal':0, 'ST':1, 'LVH':2})
db['ExerciseAngina'].unique()
db['ExerciseAngina'] = db['ExerciseAngina'].replace({'N':0, 'Y':1})
db['ST_Slope'].unique()
db['ST_Slope'] = db['ST_Slope'].replace({'Up':0, 'Down':1, 'Flat':2})
# print(db['HeartDisease'].value_counts(normalize=True))         by using (normalize=True) tells that out of 1 how many person have
#  heart disease and how many dont have. 
# db['HeartDisease'].value_counts(normalize=True).plot(kind='bar')
# plt.show() #this shows the graph
# sns.catplot(x='Sex',y='HeartDisease', data=db, kind='bar')
# plt.show()
# db['HeartDisease'].value_counts()
X = db.drop(columns='HeartDisease', axis=1)
Y = db['HeartDisease']
# print(X)
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=5)  # By using random state we can produce the same 
# result every time the code is run.
# print(X.shape, X_train.shape, X_test.shape)  
#Logistics Regression is being implemented
model = LogisticRegression(class_weight='balanced') # Balanced is used to take the minor class first and balance the dataset 
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score using ligistics regression is : ', training_data_accuracy*100,'%')
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score using ligistics regression is : ',test_data_accuracy*100,'%')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
#The dataset is adjusted to make sure that decison tree can be implemented. As the dataset is small so the dataset can be overfitted
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
model.fit(X_train, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score using decision tree is : ',test_data_accuracy*100,'%')
print('Accuracy score using decision tree is : ',training_data_accuracy*100,'%')