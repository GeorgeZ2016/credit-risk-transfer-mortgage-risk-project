# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:04:45 2017

@author: George
"""

import pandas as pd
import datetime
# Initial variables
bi,mi,t = 1000000000,1000000,1000
bal=15*bi
bal_a,bal_m1,bal_m2,bal_m3,bal_b=bal*0.95,bal*0.015,bal*0.015,bal*0.015,bal*0.005
bal_m1i, bal_m2i, bal_m1f, bal_m2f, bal_m3i, bal_m3f = bal_m1, bal_m2, bal_m1, bal_m2,bal_m3, bal_m3
# bal_a/(bal_a + bal_m1 + bal_m2 + bal_b)
years=10
start='2017-09-20'

#################################
# data for the test
index = pd.date_range(start, periods=12*years+1, freq='M')
index = index-datetime.timedelta(days = 5)

r=pd.Series(0.035/12,index=index)

def CRT(bal_a, bal_m1, bal_m2, bal_b, bal_m1i, bal_m2i, bal_m1f, bal_m2f, bal_m3, bal_m3i,
    bal_m3f, r, years, start, cdr, vpr, libor):
    # initialize each tranches
    index = pd.date_range(start, periods=12*years+1, freq='M')
    index = index-datetime.timedelta(days = 5)
    columns=['balance', 'principle', 'prepayment','loss','default','interest']
    # Output variables  
    a=pd.DataFrame(0,index=index,columns=columns)
    m1=pd.DataFrame(0,index=index,columns=columns)
    m1f=pd.DataFrame(0,index=index,columns=columns)
    m1i=pd.DataFrame(0,index=index,columns=columns)  
    m2=pd.DataFrame(0,index=index,columns=columns)
    m2i=pd.DataFrame(0,index=index,columns=columns)
    m2f=pd.DataFrame(0,index=index,columns=columns)
    m3=pd.DataFrame(0,index=index,columns=columns)
    m3i=pd.DataFrame(0,index=index,columns=columns)
    m3f=pd.DataFrame(0,index=index,columns=columns)  
    m12=pd.DataFrame(0,index=index,columns=columns) 
    ma=pd.DataFrame(0,index=index,columns=columns)     
    b=pd.DataFrame(0,index=index,columns=columns)
    t=pd.DataFrame(0,index=index,columns=columns)
    x=pd.Series(0,index=index)    
    # Variables for calculations
    total=pd.Series(0.0,index=index)
    default=pd.Series(0.0,index=index)
    loss=pd.Series(0,index=index)
    prepayment=pd.Series(0,index=index)
    principle=pd.Series(0,index=index)
    # interest
    m1_i=pd.Series(0.0,index=index)
    m1i_i =pd.Series(0.0,index=index)
    m1f_i =pd.Series(0.0,index=index)
    m2_i =pd.Series(0.0,index=index)
    m2i_i =pd.Series(0.0,index=index)
    m2f_i =pd.Series(0.0,index=index)
    m3_i =pd.Series(0.0,index=index)
    m3i_i =pd.Series(0.0,index=index)
    m3f_i=pd.Series(0.0,index=index)
    m12_i =pd.Series(0.0,index=index)
    ma_i =pd.Series(0.0,index=index)
    b_i =pd.Series(0.0,index=index)
       
    # initail balance
    a.iloc[0,0], m1.iloc[0,0], m2.iloc[0,0], b.iloc[0,0] = bal_a, bal_m1, bal_m2, bal_b
    total_bal = bal_a+bal_m1+bal_m2+bal_b+bal_m1i+bal_m2i+bal_m1f+bal_m2f+bal_m3+bal_m3i+bal_m3f 
    m1i.iloc[0,0], m2i.iloc[0,0], m1f.iloc[0,0], m2f.iloc[0,0]= bal_m1i, bal_m2i, bal_m1f, bal_m2f
    m3.iloc[0,0], m3i.iloc[0,0], m3f.iloc[0,0] = bal_m3, bal_m3i, bal_m3f 
    t.iloc[0,0]=total_bal
    g1 = total_bal*0.01
    g2 = total_bal*0.02
    
    for i in range(1,len(a)):
        # total balance of last period
        total[i] = a.iloc[i-1,0]+m1.iloc[i-1,0]+m2.iloc[i-1,0]+m3.iloc[i-1,0]+b.iloc[i-1,0]
          # prepayment(vpr or r?)
        a.iloc[i,2] = vpr[i]*total[i]*0.94
          # principle:(Here we make use of x)
        a.iloc[i,1] = (r[i]*(total[i]/(1-(1/(1+r[i])**360)))- r[i]*total[i])*0.95
          # balance:
        a.iloc[i,0] = a.iloc[i-1,0]- (a.iloc[i,1]+a.iloc[i,2])
        m1.iloc[i,0] = m1.iloc[i-1,0]- (m1.iloc[i,1]+m1.iloc[i,2])
        m2.iloc[i,0] = m2.iloc[i-1,0]- (m2.iloc[i,1]+m2.iloc[i,2])
        m3.iloc[i,0] = m3.iloc[i-1,0]- (m3.iloc[i,1]+m3.iloc[i,2])
        b.iloc[i,0] = b.iloc[i-1,0]- (b.iloc[i,1]+b.iloc[i,2])
###################################################################################      
        # interest
        b_i[i]= 0.139419*b.iloc[i-1,0]*(12.75+libor[i])/(100*12)
        m3i_i[i] = 0.557678*m3.iloc[i-1,0]*(0.75)/(100*12)
        m3f_i[i] = 0.557678*m3.iloc[i-1,0]*(5.35+libor[i])/(100*12)
        m3_i[i] = 0.557678*m3.iloc[i-1,0]*(6.35+libor[i])/(100*12)   
        m2i_i[i] = 0.557678*m2.iloc[i-1,0]*(1)/(100*12)
        m2f_i[i] = 0.557678*m2.iloc[i-1,0]*(2.00+libor[i])/(100*12)
        m2_i[i] = 0.557678*m2.iloc[i-1,0]*(2.75+libor[i])/(100*12)
        m1i_i[i] = 0.557678*m1.iloc[i-1,0]*(0.5)/(100*12)
        m1f_i[i] = 0.557678*m1.iloc[i-1,0]*(1.25+libor[i])/(100*12)
        m1_i[i] = 0.557678*m1.iloc[i-1,0]*(1.75+libor[i])/(100*12)
        
        m12_i[i] = (m1.iloc[i-1,0]+m2.iloc[i-1,0])*((1.75+libor[i])/(100*12)*(m1.iloc[i-1,0
        ]/(m1.iloc[i-1,0]+m2.iloc[i-1,0]))+(2.75+libor[i])/(100*12)*(m2.iloc[i-1,0]/(m1.iloc[
        i-1,0]+m2.iloc[i-1,0])))
        
        ma_i[i] = (m1.iloc[i-1,0]+m2.iloc[i-1,0]+m3.iloc[i-1,0])*((1.75+libor[i])/(100*12)*(
        m1.iloc[i-1,0]/(m1.iloc[i-1,0]+m2.iloc[i-1,0]+m3.iloc[i-1,0]))+(2.75+libor[i])/(100*12
        )*(m2.iloc[i-1,0]/(m1.iloc[i-1,0]+m2.iloc[i-1,0]+m3.iloc[i-1,0]))+(6.35+libor[i])/(
        100*12)*(m2.iloc[i-1,0]/(m1.iloc[i-1,0]+m2.iloc[i-1,0]+m3.iloc[i-1,0])))
        
       
###################################################################################
        # loss calculation
        default[i]=cdr[i]*total[i]        
        if sum(default)<g1:
            loss[i] = 0.15*default[i]
        if sum(default)>g1 and sum(default)<g2:
            loss[i] = 0.25*default[i]
        if sum(default)>g2:
            loss[i] = 0.40*default[i]
        
        # loss
        if b.iloc[i,0]>0:
            b.iloc[i,3] = loss[i]
            m3.iloc[i,3] = max(0, loss[i]-b.iloc[i,0])
        elif m3.iloc[i,0]>0:
            m3.iloc[i,3] = loss[i]
            m2.iloc[i,3] = max(0, loss[i]-m3.iloc[i,0])
        elif m2.iloc[i,0]>0:
            m2.iloc[i,3] = loss[i]
            m1.iloc[i,3] = max(0, loss[i]-m2.iloc[i,0])
        elif m1.iloc[i,0]>0:
            m1.iloc[i,3] = loss[i]
            a.iloc[i,3] = max(0, loss[i]-m1.iloc[i,0])
        elif a.iloc[i,0]>0:
            a.iloc[i,3] = loss[i]

###################################################################################
        # prepayment and principle
        prepayment[i]=vpr[i]*total[i]*0.05
        principle[i]=(r[i]*(total[i]/(1-(1/(1+r[i])**360)))- r[i]*total[i])*0.05

        if m1.iloc[i,0]>0:
            m1.iloc[i,2] = prepayment[i]
            m1.iloc[i,1] = principle[i]
            m2.iloc[i,1] = max(0, prepayment[i]+principle[i]-m1.iloc[i,0])
        elif m2.iloc[i,0]>0:
            m2.iloc[i,2] = prepayment[i]
            m2.iloc[i,1] = principle[i]
            m3.iloc[i,1] = max(0, prepayment[i]+principle[i]-m2.iloc[i,0])
        elif m3.iloc[i,0]>0:
            m3.iloc[i,2] = prepayment[i]
            m3.iloc[i,1] = principle[i]
            b.iloc[i,1] = max(0, prepayment[i]+principle[i]-m3.iloc[i,0])
        elif b.iloc[i,0]>0:
            b.iloc[i,2] = prepayment[i]
            b.iloc[i,1] = principle[i]        

##################################################################################   
        # Balance
        a.iloc[i,0] = max(a.iloc[i-1,0]- (a.iloc[i,1]+a.iloc[i,2]+a.iloc[i,3])
                            ,0)
        m1.iloc[i,0] = max(m1.iloc[i-1,0]- (m1.iloc[i,1]+m1.iloc[i,2]+m1.iloc[i,3])
                            ,0)
        m2.iloc[i,0] = max(m2.iloc[i-1,0]- (m2.iloc[i,1]+m2.iloc[i,2]+m2.iloc[i,3])
                            ,0)
        m3.iloc[i,0] = max(m3.iloc[i-1,0]- (m3.iloc[i,1]+m3.iloc[i,2]+m3.iloc[i,3])
                            ,0)
        b.iloc[i,0] = max(b.iloc[i-1,0]- (b.iloc[i,1]+b.iloc[i,2]+b.iloc[i,3])
                            ,0)
        
        # total data       
        t.iloc[i,0]=a.iloc[i,0]+m1.iloc[i,0]/0.557678+m2.iloc[i,0]/0.557678+m3.iloc[i,0]/0.557678+b.iloc[i,0]/0.139419
        t.iloc[i,1]=a.iloc[i,1]+m1.iloc[i,1]/0.557678+m2.iloc[i,1]/0.557678+m3.iloc[i,1]/0.557678+b.iloc[i,1]/0.139419
        t.iloc[i,2]=a.iloc[i,2]+m1.iloc[i,2]/0.557678+m2.iloc[i,2]/0.557678+m3.iloc[i,2]/0.557678+b.iloc[i,2]/0.139419
        t.iloc[i,3]=a.iloc[i,3]+m1.iloc[i,3]/0.557678+m2.iloc[i,3]/0.557678+m3.iloc[i,3]/0.557678+b.iloc[i,3]/0.139419
        t.iloc[i,4]=a.iloc[i,4]+m1.iloc[i,4]/0.557678+m2.iloc[i,4]/0.557678+m2.iloc[i,4]/0.557678+b.iloc[i,4]/0.139419
        t.iloc[i,5]=m1_i[i]+m2_i[i]+m3_i[i]+b_i[i]
        # Allocation on principle
        
        t['payment_all']=t['principle']+t['prepayment']
        t['pct']=a['principle']/t['principle']
        x[i]=a.iloc[i-1,0]+a.iloc[i,1]+a.iloc[i,2]-0.97*(t.iloc[i-1,0]+t.iloc[i,1]+
        t.iloc[i,2]-t.iloc[i,3])
       
#############################################################################        
    # 11 balance    
    m1f['balance'], m1i['balance']=m1['balance'], m1['balance']    
    m2f['balance'], m2i['balance']=m2['balance'], m2['balance']  
    m3f['balance'], m3i['balance']=m3['balance'], m3['balance']  
    m12['balance']=m1['balance']+m2['balance']
    ma['balance']=m1['balance']+m2['balance']+m3['balance']
    # 11 principle
    m1f['principle'], m1i['principle']=m1['principle'], m1['principle']    
    m2f['principle'], m2i['principle']=m2['principle'], m2['principle']  
    m3f['principle'], m3i['principle']=m3['principle'], m3['principle']  
    m12['principle']=m1['principle']+m2['principle']
    ma['principle']=m1['principle']+m2['principle']+m3['principle']
    # 11 prepayment
    m1f['prepayment'], m1i['prepayment']=m1['prepayment'], m1['prepayment']    
    m2f['prepayment'], m2i['prepayment']=m2['prepayment'], m2['prepayment']  
    m3f['prepayment'], m3i['prepayment']=m3['prepayment'], m3['prepayment']  
    m12['prepayment']=m1['prepayment']+m2['prepayment']
    ma['prepayment']=m1['prepayment']+m2['prepayment']+m3['prepayment']
    # Adding interest
    m1f['interest'], m1i['interest'], m1['interest'] = m1f_i, m1i_i, m1_i
    m2f['interest'], m2i['interest'], m2['interest'] = m2f_i, m2i_i, m2_i
    m3f['interest'], m3i['interest'], m3['interest'] = m3f_i, m3i_i, m3_i
    ma['interest'], m12['interest'], b['interest']= ma_i, m12_i, b_i
    
#################################################################################        
        
    return(a, m1, m1f, m1i, m2, m2f, m2i, m3, m3f, m3i, m12, ma, b, t, x)
 
    
############################################  

from random import gauss
def brown(n,ini):
    s=pd.Series(ini)
    for i in range(1,n):
        s[i]=s[i-1]+0.05+0.1*gauss(0,1)
    s=1+(s-1)/12
    return(s)


import crtf

#total_pv=[]
m1_pv=[]
m12_pv=[]
m1f_pv=[]
m1i_pv=[]
m2_pv=[]
m2f_pv=[]
m2i_pv=[]
m3_pv=[]
m3f_pv=[]
m3i_pv=[]
ma_pv=[]
b_pv=[]
g_pv=[]
ratio=(1-0.219556)
i=0
while i<500:
    vpr_pre, cdr_pre, libor_f=crtf.simulate()
    l=len(index)
    vpr_pre, cdr_pre, libor_f=vpr_pre[:l]-2, cdr_pre[:l], libor_f[:l]
    tr_a, tr_m1, tr_m1f, tr_m1i, tr_m2, tr_m2f, tr_m2i, tr_m3, tr_m3f, tr_m3i, tr_m12, tr_ma, tr_b, total, x = CRT(bal_a, bal_m1, bal_m2, bal_b, bal_m1i, bal_m2i, bal_m1f, bal_m2f, 
                    bal_m3, bal_m3i, bal_m3f, r, years, start, cdr_pre/1200, vpr_pre/1200,
                    libor_f)
    lib=libor_f.cumprod()
    #v=(total['principle']+total['interest'])/lib.values
    m1=sum((tr_m1['interest']/lib.values).dropna())+sum(
            ((tr_m1['principle']+tr_m1['prepayment'])[tr_m1['balance'
              ]!=0]*ratio/lib[:sum(tr_m1['balance']!=0)].values).dropna())
    m12=sum((tr_m12['interest']/lib.values).dropna())+sum(
            ((tr_m12['principle']+tr_m12['prepayment'])[tr_m12['balance'
              ]!=0]*ratio/lib[:sum(tr_m12['balance']!=0)].values).dropna())
    m1f=sum((tr_m1f['interest']/lib.values).dropna())+sum(
            ((tr_m1f['principle']+tr_m1f['prepayment'])[tr_m1f['balance'
              ]!=0]*ratio/lib[:sum(tr_m1f['balance']!=0)].values).dropna())
    m1i=sum((tr_m1i['interest']/lib.values).dropna())+sum(
            ((tr_m1i['principle']+tr_m1i['prepayment'])[tr_m1i['balance'
              ]!=0]*ratio/lib[:sum(tr_m1i['balance']!=0)].values).dropna())
    m2=sum((tr_m2['interest']/lib.values).dropna())+sum(
            ((tr_m2['principle']+tr_m2['prepayment'])[tr_m2['balance'
              ]!=0]*ratio/lib[:sum(tr_m2['balance']!=0)].values).dropna())
    m2f=sum((tr_m2f['interest']/lib.values).dropna())+sum(
            ((tr_m2f['principle']+tr_m2f['prepayment'])[tr_m2f['balance'
              ]!=0]*ratio/lib[:sum(tr_m2f['balance']!=0)].values).dropna())
    m2i=sum((tr_m2i['interest']/lib.values).dropna())+sum(
            ((tr_m2i['principle']+tr_m2i['prepayment'])[tr_m2i['balance'
              ]!=0]*ratio/lib[:sum(tr_m2i['balance']!=0)].values).dropna())
    m3=sum((tr_m3['interest']/lib.values).dropna())+sum(
            ((tr_m3['principle']+tr_m3['prepayment'])[tr_m3['balance'
              ]!=0]*ratio/lib[:sum(tr_m3['balance']!=0)].values).dropna())
    m3f=sum((tr_m3f['interest']/lib.values).dropna())+sum(
            ((tr_m3f['principle']+tr_m3f['prepayment'])[tr_m3f['balance'
              ]!=0]*ratio/lib[:sum(tr_m3f['balance']!=0)].values).dropna())
    m3i=sum((tr_m3i['interest']/lib.values).dropna())+sum(
            ((tr_m3i['principle']+tr_m3i['prepayment'])[tr_m3i['balance'
              ]!=0]*ratio/lib[:sum(tr_m3i['balance']!=0)].values).dropna())
    ma=sum((tr_ma['interest']/lib.values).dropna())+sum(
            ((tr_ma['principle']+tr_ma['prepayment'])[tr_ma['balance'
              ]!=0]*ratio/lib[:sum(tr_ma['balance']!=0)].values).dropna())
    b=sum((tr_b['interest']/lib.values).dropna())+sum(
            ((tr_b['principle']+tr_b['prepayment'])[tr_b['balance'
              ]!=0]*ratio/lib[:sum(tr_b['balance']!=0)].values).dropna())
    g=sum(total['balance'].values*(0.002/12)/lib)

    m1_pv.append(m1)
    m12_pv.append(m12)
    m1f_pv.append(m1f)
    m1i_pv.append(m1i)
    m2_pv.append(m2)
    m2f_pv.append(m2f)
    m2i_pv.append(m2i)
    m3_pv.append(m3)
    m3f_pv.append(m3f)
    m3i_pv.append(m3i)
    ma_pv.append(ma)
    b_pv.append(b)
    g_pv.append(g)
    i=i+1

import numpy as np
# Mean and confidence interval
print('The PV of tranche M1 is: %.5f' %(sum(m1_pv)/len(m1_pv)))
print('The PV of tranche M12 is: %.5f' %(sum(m12_pv)/len(m12_pv)))
print('The PV of tranche M1f is: %.5f' %(sum(m1f_pv)/len(m1f_pv)))
print('The PV of tranche M1i is: %.5f' %(sum(m1i_pv)/len(m1i_pv)))
print('The PV of tranche M2 is: %.5f' %(sum(m2_pv)/len(m2_pv)))
print('The PV of tranche M2f is: %.5f' %(sum(m2f_pv)/len(m2f_pv)))
print('The PV of tranche M2i is: %.5f' %(sum(m2i_pv)/len(m2i_pv)))
print('The PV of tranche M3 is: %.5f' %(sum(m3_pv)/len(m3_pv)))
print('The PV of tranche M3f is: %.5f' %(sum(m3f_pv)/len(m3f_pv)))
print('The PV of tranche M3i is: %.5f' %(sum(m3i_pv)/len(m3i_pv)))
print('The PV of tranche MA is: %.5f' %(sum(ma_pv)/len(ma_pv)))
print('The PV of tranche B is: %.5f' %(sum(b_pv)/len(b_pv)))
print('The PV of tranche guarantee fee is: %.5f' %(sum(g_pv)/len(g_pv)))

print('The 95%% Confidence interval of tranche M1 is: (%.5f,%.5f)' %((np.mean(m1_pv)-1.96*np.std(
      m1_pv)/len(m1_pv)**0.5) ,(np.mean(m1_pv)+1.96*np.std(m1_pv)/len(m1_pv)**0.5)))
print('The 95%% Confidence interval of tranche M12 is: (%.5f,%.5f)' %((np.mean(m12_pv)-1.96*np.std(
      m12_pv)/len(m12_pv)**0.5) ,(np.mean(m12_pv)+1.96*np.std(m12_pv)/len(m12_pv)**0.5)))
print('The 95%% Confidence interval of tranche M1f is: (%.5f,%.5f)' %((np.mean(m1f_pv)-1.96*np.std(
      m1f_pv)/len(m1f_pv)**0.5) ,(np.mean(m1f_pv)+1.96*np.std(m1f_pv)/len(m1f_pv)**0.5)))
print('The 95%% Confidence interval of tranche M1i is: (%.5f,%.5f)' %((np.mean(m1i_pv)-1.96*np.std(
      m1i_pv)/len(m1i_pv)**0.5) ,(np.mean(m1i_pv)+1.96*np.std(m1i_pv)/len(m1i_pv)**0.5)))
print('The 95%% Confidence interval of tranche M2 is: (%.5f,%.5f)' %((np.mean(m2_pv)-1.96*np.std(
      m2_pv)/len(m2_pv)**0.5) ,(np.mean(m2_pv)+1.96*np.std(m2_pv)/len(m2_pv)**0.5)))
print('The 95%% Confidence interval of tranche M2f is: (%.5f,%.5f)' %((np.mean(m2f_pv)-1.96*np.std(
      m2f_pv)/len(m2f_pv)**0.5) ,(np.mean(m2f_pv)+1.96*np.std(m2f_pv)/len(m2f_pv)**0.5)))
print('The 95%% Confidence interval of tranche M2i is: (%.5f,%.5f)' %((np.mean(m2i_pv)-1.96*np.std(
      m2i_pv)/len(m2i_pv)**0.5) ,(np.mean(m2i_pv)+1.96*np.std(m2i_pv)/len(m2i_pv)**0.5)))
print('The 95%% Confidence interval of tranche M3 is: (%.5f,%.5f)' %((np.mean(m3_pv)-1.96*np.std(
      m3_pv)/len(m3_pv)**0.5) ,(np.mean(m3_pv)+1.96*np.std(m3_pv)/len(m3_pv)**0.5)))
print('The 95%% Confidence interval of tranche M3f is: (%.5f,%.5f)' %((np.mean(m3f_pv)-1.96*np.std(
      m3f_pv)/len(m3f_pv)**0.5) ,(np.mean(m3f_pv)+1.96*np.std(m3f_pv)/len(m3f_pv)**0.5)))
print('The 95%% Confidence interval of tranche M3i is: (%.5f,%.5f)' %((np.mean(m3i_pv)-1.96*np.std(
      m3i_pv)/len(m3i_pv)**0.5) ,(np.mean(m3i_pv)+1.96*np.std(m3i_pv)/len(m3i_pv)**0.5)))
print('The 95%% Confidence interval of tranche MA is: (%.5f,%.5f)' %((np.mean(ma_pv)-1.96*np.std(
      ma_pv)/len(ma_pv)**0.5) ,(np.mean(ma_pv)+1.96*np.std(ma_pv)/len(ma_pv)**0.5)))
print('The 95%% Confidence interval of tranche B is: (%.5f,%.5f)' %((np.mean(b_pv)-1.96*np.std(
      b_pv)/len(b_pv)**0.5) ,(np.mean(b_pv)+1.96*np.std(b_pv)/len(b_pv)**0.5)))
print('The 95%% Confidence interval of guarantee fee is: (%.5f,%.5f)' %((np.mean(g_pv)-1.96*np.std(
      g_pv)/len(g_pv)**0.5) ,(np.mean(g_pv)+1.96*np.std(g_pv)/len(g_pv)**0.5)))

##########################################################

t1=sum(b_pv)/len(b_pv)+sum(m1_pv)/len(m1_pv)+sum(m2_pv)/len(m2_pv)+sum(m3_pv
        )/len(m3_pv)
print('The 95%% Confidence interval of total PV is: (%.5f,%.5f)' %((t1-1.96*np.std(
      b_pv+m3_pv+m2_pv+m1_pv)/len(b_pv)**0.5) ,(t1+1.96*np.std(
      b_pv+m3_pv+m2_pv+m1_pv)/len(b_pv)**0.5)))

t2=sum(g_pv)/len(g_pv)
interest=sum(total['interest'].values/lib)

"""
base='C:/Users/George/Desktop/project/output/'
tr_a.to_csv(base+'tr_a.csv')
tr_b.to_csv(base+'tr_b.csv')
tr_m1.to_csv(base+'tr_m1.csv')
tr_m12.to_csv(base+'tr_m12.csv')
tr_m1f.to_csv(base+'tr_m1f.csv')
tr_m1i.to_csv(base+'tr_m1i.csv')
tr_m2.to_csv(base+'tr_m2.csv')
tr_m2f.to_csv(base+'tr_m2f.csv')
tr_m2i.to_csv(base+'tr_m2i.csv')
tr_m3.to_csv(base+'tr_m3.csv')
tr_m3f.to_csv(base+'tr_m3f.csv')
tr_m3i.to_csv(base+'tr_m3i.csv')
tr_ma.to_csv(base+'tr_ma.csv')
"""