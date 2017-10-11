# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:28:09 2017

@author: George
"""

from math import exp,log,sqrt
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import numpy as np

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


def brown(n,ini):
    s=pd.Series(ini)
    for i in range(1,n):
        s[i]=s[i-1]+0.05+0.1*gauss(0,1)
    s=1+(s-1)/12
    return(s)

def simulate():

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
    # DATA FOR MODEL 1
    libor.index=pd.to_datetime(libor.index)
    house.index=pd.to_datetime(house.index)
    libor=libor/100
    """
    ## features for vpr modeling
    feature1=pd.merge(libor,vpr1,how='inner',left_index=True,right_index=True).sort_index()
    ## features for cdr modeling
    feature2=pd.merge(house,cdr1,how='inner',left_index=True,right_index=True).sort_index()
    
    age1=np.linspace(1,len(feature1),len(feature1))
    # ADD AGE SQUARE
    age2=np.linspace(1,len(feature2),len(feature2))
    age_square=age2**2
    feature1=pd.merge(pd.DataFrame(age1,index=feature1.index), feature1,left_index=True,
                      right_index=True)
    feature1.columns=['age','libor','vpr']
    feature2=pd.merge(pd.DataFrame(age2,index=feature2.index), feature2,left_index=True,
                      right_index=True)
    feature2=pd.merge(pd.DataFrame(age_square,index=feature2.index), feature2,left_index=True,
                      right_index=True)
    feature2.columns=['age_square','age','cpihousing','cdr']
    """
    # DATA FOR MODEL 2
    
    feature_1=pd.merge(libor,house,how='inner',left_index=True,right_index=True).sort_index()
    feature_1=pd.merge(feature_1,vpr2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
    feature_2=pd.merge(libor,house,how='inner',left_index=True,right_index=True).sort_index()
    feature_2=pd.merge(feature_2,cdr2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
    age1=np.linspace(1,len(feature_2),len(feature_2))
    age2=np.linspace(1,len(feature_2),len(feature_2))**2
    feature_2=pd.merge(pd.DataFrame(age2,index=feature_2.index),feature_2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
    feature_2=pd.merge(pd.DataFrame(age1,index=feature_2.index),feature_2,how='inner',left_index=True,right_index=True).sort_index()['2010':]
        
    feature_1.columns=['libor','cpihousing','vpr']
    feature_2.columns=['age','age2','libor','cpihousing','cdr']
 
    feature=np.array(feature_1[1:-1])
    X_train=feature[:,:-1]
    y_train=feature[:,-1]
    model1=LinearRegression()
    model1.fit(X_train, y_train)
    #y=model1.predict(X_train)
    #pd.Series(y).plot()
    
    feature=np.array(feature_2[1:-1])
    X_train=feature[:,:-1]
    y_train=feature[:,-1]
    model2=Ridge(alpha=1)
    model2.fit(X_train, y_train)
    #y=model2.predict(X_train)
    #pd.Series(y).plot()
    
    # plot the result
    
    start='2016-03-25'
    years=12.5
    index = pd.date_range(start, periods=12*years+1, freq='M')
    
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
    data_cdr['age2']=data_cdr[0]**2
    data_cdr['libor']=data_vpr['libor'].values
    data_cdr['house']=0.05/12
    data_cdr.columns=['age','age2','libor','housing']
    
    
    feature_vpr1=np.array(data_vpr[['libor','house']])
    feature_cdr1=np.array(data_cdr[['age','age2','libor','housing']])
    vpr_pre=model1.predict(feature_vpr1)
    cdr_pre=model2.predict(feature_cdr1)
    vpr_pre[vpr_pre<0]=2
    cdr_pre[cdr_pre<0]=0.02
    cdr_pre[:5]=0
    
    return(vpr_pre, cdr_pre, libor_f)