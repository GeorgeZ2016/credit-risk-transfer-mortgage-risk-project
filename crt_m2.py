# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:29:49 2017

@author: George
"""

import pandas as pd
import numpy as np
###########################################################
"""
## Clean the modeling data
mr=pd.read_excel('C:/Users/George/Desktop/project/mortgage rate/all.xlsx'
                 ,header=1)
cdr1=pd.DataFrame(mr.iloc[1:,15].dropna())
cdr1.index=pd.to_datetime(cdr1.index)
cdr1.columns=['cdr']
cdr1['cdr']=cdr1['cdr']/1000
vpr1=pd.DataFrame(mr.iloc[1:,0].dropna())
vpr1.index=pd.to_datetime(vpr1.index)
vpr1.columns=['vpr']
vpr1['vpr']=vpr1['vpr']/1000
"""
mr=pd.read_excel('C:/Users/George/Desktop/project/mortgage rate/all.xlsx'
                 ,header=1)
data=pd.read_excel('C:/Users/George/Desktop/project/VPR-CDR.xlsx'
                 ,header=1, index_col='Date')
data=data.dropna()
data.index=pd.to_datetime(data.index)
vpr1=pd.DataFrame(data.iloc[:,0])/100
cdr1=pd.DataFrame(data.iloc[:,1])

vpr2=pd.DataFrame(mr.iloc[1:-2,0])
vpr2.index=pd.to_datetime(vpr2.index)
cdr2=pd.DataFrame(mr.iloc[1:,15].dropna())
cdr2.index=pd.to_datetime(cdr2.index)

#####################################################
# read features data

house=pd.read_csv('C:/Users/George/Desktop/project/features/cpihousing.csv', 
                  index_col='DATE')
libor=pd.read_csv('C:/Users/George/Desktop/project/features/libor.csv',
                     index_col='DATE')
house['CPIHOSNS']=house['CPIHOSNS'].diff()/house['CPIHOSNS']
libor[libor[libor.columns]=='.']=None
libor=libor.fillna(method='bfill')
libor=libor.astype('float')
libor.index.name='libor'
libor.index=pd.to_datetime(libor.index)

from pandas.tseries.offsets import Day
libor=libor.asfreq(Day()).fillna(method='ffill')

####################################################
# merge datasets

# DATA FOR MODEL 1
libor.index=pd.to_datetime(libor.index)
house.index=pd.to_datetime(house.index)
libor=libor/100
## features for vpr modeling
feature1=pd.merge(libor,vpr1,how='inner',left_index=True,right_index=True).sort_index()
## features for cdr modeling
feature2=pd.merge(house,cdr1,how='inner',left_index=True,right_index=True).sort_index()

age1=np.linspace(1,len(feature1),len(feature1))
age2=np.linspace(1,len(feature2),len(feature2))
feature1=pd.merge(pd.DataFrame(age1,index=feature1.index), feature1,left_index=True,
                  right_index=True)
feature1.columns=['age','libor','vpr']
feature2=pd.merge(pd.DataFrame(age2,index=feature2.index), feature2,left_index=True,
                  right_index=True)
feature2.columns=['age','cpihousing','cdr']

# DATA FOR MODEL 2

feature_1=pd.merge(libor,house,how='inner',left_index=True,right_index=True).sort_index()
feature_1=pd.merge(feature_1,vpr2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
feature_2=pd.merge(libor,house,how='inner',left_index=True,right_index=True).sort_index()
feature_2=pd.merge(feature_2,cdr2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
feature_1.columns=['libor','cpihousing','vpr']
feature_2.columns=['libor','cpihousing','cdr']
##################################################3
# modeling

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_validation import cross_val_score,LeaveOneOut
from sklearn.metrics import mean_squared_error
###########################
# feature data can change here


feature=np.array(feature_1[1:-1])

############################
# model 1:
loo=LeaveOneOut(len(feature))
model=Lasso(alpha=1000000000)

X_train=feature[:,:-1]
y_train=feature[:,-1]
scores = cross_val_score(estimator=model, X=X_train, y=y_train, scoring='mean_squared_error',
                         cv=loo)
print('The mean MSE is:%.10f' %scores.mean())
############################
# model 2:
loo=LeaveOneOut(len(feature))
model=Ridge(alpha=1)

X_train=feature[:,:-1]
y_train=feature[:,-1]
scores = cross_val_score(estimator=model, X=X_train, y=y_train, scoring='mean_squared_error',
                         cv=loo)
print('The mean MSE is:%.10f' %scores.mean())
############################
# model 3:
loo=LeaveOneOut(len(feature))
model=LinearRegression()

X_train=feature[:,:-1]
y_train=feature[:,-1]
scores = cross_val_score(estimator=model, X=X_train, y=y_train, scoring='mean_squared_error',
                         cv=loo)
print('The mean MSE is:%.10f' %scores.mean())

################################################

feature=np.array(feature1[1:-1])
X_train=feature[:,:-1]
y_train=feature[:,-1]
model1=LinearRegression()
model1.fit(X_train, y_train)
y_plot=model1.predict(X_train)

feature=np.array(feature2[1:-1])
X_train=feature[:,:-1]
y_train=feature[:,-1]
model2=Ridge(alpha=1)
model2.fit(X_train, y_train)
y2_plot=model2.predict(X_train)

# plot the result
y_p=pd.DataFrame(y_plot,index=feature1.index[1:-1],columns=['regression output'])
y_p.sort_index().plot(title='vpr and ridge regression',legend=True)
feature1.sort_index()['vpr'][1:-1].plot(legend=True)
y2_p=pd.DataFrame(y2_plot,index=feature2.index[1:-1],columns=['regression output'])
y2_p.sort_index().plot(title='cdr and simple linear regression',legend=True)
feature2.sort_index()['cdr'][1:-1].plot(legend=True)

#########################################################

start='2016-03-25'
years=12.5
index = pd.date_range(start, periods=12*years+1, freq='M')

############################
# vacisek part(useless)
#############################
from math import exp,log,sqrt
def calibrate_ml(S,delta=1):

    n = len(S)-1
    Sx=sum( S[:-1] )
    Sy=sum( S[1:] )
    Sxx=sum( S[:-1]**2 )
    Sxy=sum( (S[:-1]*S[1:]).dropna())
    Syy=sum( S[1:]**2 )
     
    mu=(Sy*Sxx-Sx*Sxy)/(n*(Sxx-Sxy)-(Sx**2-Sx*Sy))
    lambd=-log((Sxy-mu*Sx-mu*Sy+n*mu**2)/(Sxx-2*mu*Sx+n*mu**2))/delta
    a=exp(-lambd*delta)
    sigmah2 = (Syy-2*a*Sxy+(a**2)*Sxx-2*mu*(1-a)*(Sy-a*Sx)+n*(mu**2)*((1-a)**2))/n
    sig =sqrt((sigmah2*2)*lambd/(1-(a**2)))
    return(lambd,mu,sig)
#######################
from random import gauss
def myVasicek(K,theta,sigma,r0,T,N):
    t=T/N
    r=pd.Series(r0)

    for i in range(1,N):
        r[i]=r[i-1]*exp(-K*t)+theta*(1-exp(-K*t))+sigma*sqrt((1-exp(-2*K*t))/(2*K))*gauss(0,1)
    return(r)
############################################
"""
model.fit(X_train, y_train)
l=model.coef_
model.intercept_
"""

def brown(n,ini):
    s=pd.Series(ini)
    for i in range(1,n):
        s[i]=s[i-1]+0.05+0.1*gauss(0,1)
    s=1+(s-1)/12
    return(s)
libor1=brown(5*12,1.22389)
    
delta=1
t=12*years+1-5*12+1
lambd,mu,sig=calibrate_ml(pd.Series(libor.iloc[:,0]))
libor_ft=myVasicek(lambd,mu,sig,libor1[len(libor1)-1],t,int(t/delta))
libor_f=pd.concat( [ libor1, libor_ft[1:] ], ignore_index=True)


age=np.linspace(1,len(index),len(index))
data_vpr=pd.DataFrame(age,index=index)
data_vpr['libor']=libor_f.values/100
data_vpr.columns=['age','libor']
data_vpr['house']=0.05/12
data_cdr=pd.DataFrame(age,index=index)
data_cdr['libor']=data_vpr['libor'].values
data_cdr['house']=0.05/12
data_cdr.columns=['age','libor','housing']


feature_vpr1=np.array(data_vpr[['age','libor']])
feature_cdr1=np.array(data_cdr[['age','housing']])
vpr_pre=model1.predict(feature_vpr1)
cdr_pre=model2.predict(feature_cdr1)

#vpr_pre=vpr_pre+0.02
vpr_pre[vpr_pre<0]=0
#pd.Series(vpr_pre).plot()
#pd.Series(cdr_pre).plot()

###########################################
# Model 2
feature=np.array(feature_1)
X_train=feature[:,:-1]
y_train=feature[:,-1]
model_1=LinearRegression()
model_1.fit(X_train, y_train)
vpr_plot=model_1.predict(X_train)

feature=np.array(feature_2)
X_train=feature[:,:-1]
y_train=feature[:,-1]
model_2=LinearRegression()
model_2.fit(X_train, y_train)
cdr_plot=model_2.predict(X_train)

# result
y_p=pd.DataFrame(vpr_plot,index=feature_1.index,columns=['regression output'])
y2_p=pd.DataFrame(cdr_plot,index=feature_2.index,columns=['regression output'])



feature_vpr2=np.array(data_vpr[['libor','house']])
feature_cdr2=np.array(data_cdr[['libor','housing']])
vpr_pre=model_1.predict(feature_vpr2)
cdr_pre=model_2.predict(feature_cdr2)
"""
# Plot the result
y_p.sort_index().plot(title='vpr and linear regression',legend=True)
feature_1.sort_index()['vpr'].plot(legend=True)
y2_p.sort_index().plot(title='cdr and linear regression',legend=True)
feature_2.sort_index()['cdr'].plot(legend=True)
pd.Series(vpr_pre).plot()
pd.Series(cdr_pre).plot()
"""
##############################################################################


