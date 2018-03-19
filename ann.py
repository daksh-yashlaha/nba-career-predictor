# ANN Project on prediction if rookie NBA players will last more than 5 yrs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense

#Importing Data-set from csv file
dataset = pd.read_csv("nba_logreg.csv", header = 0)
X = dataset.iloc[: , 1:20].values
y = dataset.iloc[:, 20].values

#Data Pre-Processing
#Taking care of Missing Values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

#Splitting dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

#Standardizing the data
stdscaler = StandardScaler()
X_train = stdscaler.fit_transform(X_train)
X_test = stdscaler.transform(X_test)

#Making ANN model
#Making first layer and first hidden layer
classifier = Sequential()
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', 
                     input_dim = 19))

#Making second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

#Making third hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

#Making fourth hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

#Making output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compilation of ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 500)

#Predicting the test set values
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Making confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

#Making classification report
cr = classification_report(y_test, y_pred)
print("Classification Report")
print(cr)

#Making the crosstab
y_pred = np.squeeze(y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], 
            margins=True)


#Plotting graphs
#Plot 1
plt.xlabel("No. of matches played")
plt.ylabel("No. of players with career greater than 5 yrs")
plt.title("Data Analysis")

k=-1
u=np.linspace(10,100,10)
z = np.zeros(10)
for i in X[:,0]:
    k+=1
    for j in range(0,10):
            if(((j*10)<=i)and(i<=((j+1)*10))and((y[k])==1)):
                z[j]+=1
plt.bar(u,z,color='blue')
plt.show()

#Plot 2
plt.xlabel("Points Scored per Game")
plt.ylabel("No. of players with career greater than 5 yrs")
plt.title("Data Analysis")

k=-1
u=np.linspace(0,30,11)
z = np.zeros(11)
for i in X[:,2]:
    k+=1
    for j in range(0,10):
            if(((j*3)<=i)and(i<=((j+1)*3))and((y[k])==1)):
                z[j]+=1
plt.bar(u,z,color='red')
plt.show()

#Plot 3 Confuison Matrix
plt.title("Confusion Matrix")
label = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
values = [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
plt.pie(values, labels = label, shadow = True, autopct = '%.0f%%')
plt.show()
