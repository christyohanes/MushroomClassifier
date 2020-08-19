# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:20:25 2020

@author: yohanes
"""


# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:32:27 2020

@author: yohanes
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

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

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X, Y)
features = fit.transform(X)
np.set_printoptions(precision=6) 
print(fit.scores_)