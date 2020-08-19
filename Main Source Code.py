# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:32:27 2020

@author: yohanes
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

direktori="MUSHROOMS.csv"

names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises','odor','gill-attachment','gill-spacing','gill-size'
         ,'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring'
         ,'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

dataset = pd.read_csv(direktori, names=names)

#baca dataset
arrayData = dataset.values

#mengisi data yang (?)
#imputasi data
x = arrayData[:,:] 
x[x=='?']='nan'
imp = SimpleImputer(missing_values='nan', strategy="most_frequent")
x = imp.fit_transform(x)
data = pd.DataFrame(x,columns = names)

#membuat labelEncoder : mengubah varible dan kelas string ke integer (sesuai urutan huruf) 
le = preprocessing.LabelEncoder()
data_en=data.copy()
for i in data_en.columns:
    data_en[i] = le.fit_transform(data_en[i])
    
arrayDF = data_en.values

#Memisahkan fitur dan kelas
X = arrayDF[:,1:23]  #fitur
Y = arrayDF[:,0]     #kelas

X = np.delete(X, [2,3,10,13,14,15], 1)

#normalisasi min max range 0 sampai 1
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

#variable untuk menyimpan kelas hasil testing dan kelas asli
prediksi=list()
harapan=list()

#teknik pembagian data 10-Fold Cross Validation 
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold (n_splits=10)
for train, test in skf.split(rescaledX ,Y):
    x_train, x_test, y_train, y_test = rescaledX[train], rescaledX[test], Y[train], Y[test]

#create a KNN classifier
model = KNeighborsClassifier(n_neighbors=3)
    
#train the model using the training sets
model.fit(x_train, y_train)

#predict Output
predicted = model.predict(x_test)
       
#menyimpan hasil prediksi dan harapannya ke dalam variable 
prediksi.append(predicted)
harapan.append(y_test)
    
#menggabungkan isi prediksi dan harapan
import itertools
pred=list(itertools.chain.from_iterable(prediksi))
harap=list(itertools.chain.from_iterable(harapan))
    
#Pembuatan confussion matrices
def Conf_matrix(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)) :
        if y_actual[i] == y_pred[i] == 1 :
            TP += 1
        if y_pred[i]== 1 and y_actual[i] != y_pred[i] :
            FP += 1
        if y_actual[i] == y_pred[i] == 0 :
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i] :
            FN += 1  
    return(TP, FN, TN, FP)
    
TP, FN, TN, FP=Conf_matrix(y_test,predicted)

print("10-Fold Cross Validation")
print()

print('akurasi = ',(((TP+TN)/(TP+TN+FP+FN))*100))
print('sensitivity = ',((TP/(TP+FN))*100))
print('specificity = ',((TN/(TN+FP))*100))
print()


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, predicted))  
print(classification_report(y_test, predicted))

# Visualize and Calculating error for K values between 1 and 40
import matplotlib.pyplot as plt 
error = []
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate Nilai K')  
plt.xlabel('Nilai K')  
plt.ylabel('Error rata-rata')
