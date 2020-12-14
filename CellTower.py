#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:50:44 2018

@author: deola
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Reading and importing 
the excel file into our workspace
                              '''
dataset=pd.ExcelFile("/home/deola/Documents/Opti-Num/20121001(1).xlsx")
capacity2Matrix=pd.read_excel(dataset, "capacity2Matrix",  header=None)
cellsVec=pd.read_excel(dataset, "cellsVec",  header=None)
erlangCombMatrix=pd.read_excel(dataset, "erlangCombMatrix",  header=None)
timestampVec=pd.read_excel(dataset, "timestampVec", header=None)




'''Take the element in the timestampVec
   and put it in an array'''
timeArr=list()
df=pd.DataFrame(timestampVec)
for m in df.values[0:len(df),0]:
    timeArr.append(m.encode('ascii', 'ignore'))
    
#timeArr=[str(r) for r in timeArr]
#print timeArr


df1=pd.DataFrame(cellsVec)
#print(df[0])
erlangCombMatrix.columns=timeArr
df2=pd.DataFrame(erlangCombMatrix)
df_NewErlangComb=pd.concat([df1,df2], axis=1)


capacity2Matrix.columns=timeArr
df3=pd.DataFrame(capacity2Matrix)
df_NewCapacity=pd.concat([df1,df3], axis=1)
#df_NewCapacity=pd.concat([df.T,df_NewCapacity])

df_NewCapacity.to_csv("/home/deola/Documents/Opti-Num/capacity.csv", index=None)
df_NewErlangComb.to_csv("/home/deola/Documents/Opti-Num/erlangComb.csv", index=None)

#print(df_NewCapacity.loc["1",:])

data=pd.read_csv("/home/deola/Documents/Opti-Num/capacity.csv")
data2=pd.read_csv("/home/deola/Documents/Opti-Num/erlangComb.csv")
data=data.values
data2=data2.values
cellsVecArr=data[0:len(data),0]
groupDict=dict()
row,column=data.shape



def check_cell_with_no_capacity():
    
    '''List cellsVecArr without 
        network presence They are 62 in number.'''  
    
    #print cellsVecArr
    noNetwork=list()
    
    for i in range(1, column):
        X1=data[0:len(data),i]
        X2=data2[0:len(data2),i]
        
        for j in range (0,len(X1)):
            value=X1[j]
            if value==0:
                noNetwork.append(cellsVecArr[j])
    noNetwork=list(set(noNetwork))
    print noNetwork
    print len(noNetwork)
            


def check_exceed_capacity():
    
    '''Code to add and check exceed capacity:
    Cell that exceed the maxium capacity
     are 6322 in number.
    '''

    cellVecExceedingCapacity=list()
    
    for i in range(1, column):
        X1=data[0:len(data),i]
        X2=data2[0:len(data2),i]
        
        for j in range (0,len(X1)):
            value=X1[j]
            if value!=0:
                percentage=X2[j]*100/value
                #print percentage
                if percentage>100:
                     cellVecExceedingCapacity.append(cellsVecArr[j])
    cellVecExceedingCapacity=list(set(cellVecExceedingCapacity))
    print cellVecExceedingCapacity
    print len(cellVecExceedingCapacity)



def grouping(dis=50, noDis=85):
        
    '''
    Code to: 
    Group cellVec into three
       Discount, No discount and Higher tariff.
       
    For our default settings, when the discount is less than 50%, we issue discount
    when it is greater than 50% and less than 85% we issue no discount. When it is 
    greater than 85% we issue high tariff.
    '''

    discountArr,noDiscountArr, higherTarrifArr=list(), list(), list()
    global groupDict
    
    for i in range(1, column):
        X1=data[0:len(data),i]
        X2=data2[0:len(data2),i]
        
        for j in range (0,len(X1)):
            value=X1[j]
            if value!=0:
                percentage=X2[j]*100/value
                #print percentage
                if percentage<dis:
                    discountArr.append(cellsVecArr[j])
                elif percentage>dis and percentage<noDis:
                    noDiscountArr.append(cellsVecArr[j])
                elif percentage>=noDis:
                    higherTarrifArr.append(cellsVecArr[j])
        
        group=pd.DataFrame(discountArr, columns=['discount'])
        group["noDiscount"]=pd.DataFrame(noDiscountArr)
        group["higherTarrif"]=pd.DataFrame(higherTarrifArr)
        group.to_csv("/home/deola/Documents/Opti-Num/result/group_%sHour.csv"%str(i-1),index=None)
        
        groupDict["cellgroup%s"%str(i-1)]=[len(discountArr), len(noDiscountArr), len(higherTarrifArr) ]
        
        discountArr,noDiscountArr, higherTarrifArr=list(), list(), list()         
    print  groupDict



def plot_pie_chart():
    '''Plots the pie chart for cells due
       for discount, no discount and higher tariff per hour.
       
       Call function GROUPING before calling this function.
       '''
    
    groupHeader='discount',"noDiscount", "higherTarrif"
    
    
    for key,value in groupDict.items():
        
        fig1, ax1 = plt.subplots()
        ax1.pie(value, labels=groupHeader)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title("A pie-chart of cellgroup by the %s:00Hr"%str(key[9:]))
        plt.savefig("/home/deola/Documents/Opti-Num/result/pie_%s.eps"%str(key), format="eps", dpi=300)

    #plt.show()
def reading_rem():
    '''Importing the rest of the dataset
        Into our workspace'''
    Cell=list()
    
    data_Dur=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetDurMatrix.csv", sep='\t', engine='python')
    data_Dur=pd.DataFrame(data_Dur.T)
    data_Dur.to_csv("/home/deola/Documents/Opti-Num/20121001_onnetDurMatrix1.csv")
    
    data_Disc=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetDiscMatrix.csv", sep='\t', engine='python')
    data_Disc=pd.DataFrame(data_Disc.T)
    data_Disc.to_csv("/home/deola/Documents/Opti-Num/20121001_onnetDiscMatrix1.csv")
    
    data_Rev=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix.csv", sep='\t', engine='python')
    data_Rev=pd.DataFrame(data_Rev.T)
    #data.shape
    data_Rev.to_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix1.csv")
    
    
    
    data_Cell_Class=pd.read_excel("/home/deola/Documents/Opti-Num/cellClassificationData.xlsx")
    data_Cell_Class =data_Cell_Class.values
    comp_Cell=data_Cell_Class[0:len(data_Cell_Class), 0]
    
    
    for m in comp_Cell:
        Cell.append(m.encode('ascii', 'ignore'))
    print( "'9829C'" in Cell)
#############################################################################################
def data_visual():
     from pandas.tools.plotting import autocorrelation_plot
    
     
     data_Rev=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix1.csv" )
     row, column=data_Rev.shape
     data_Rev.columns=timeArr
     ARR=list()
     for m in timeArr:
         ARR.append(m[12:].replace("'", ""))
     #print ARR
     data_Rev.columns=ARR
     Row1=data_Rev.loc[[15]]
     Row1 =pd.DataFrame(Row1)
    
     Row1.T.to_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix2.csv", header=None)
     valuesRev=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix2.csv", names=["time","Rev"])
     valuesRev=valuesRev.values
     X=valuesRev[0:row, 1]
     print X
     plt.plot(ARR, X)
     plt.xticks(ARR, rotation='vertical')
     plt.tight_layout(pad=0, h_pad=0, w_pad=0)
     plt.savefig("/home/deola/Documents/Opti-Num/result/trend.eps", format="eps", dpi=300)

     #m.plot()
     #autocorrelation_plot(m)
     #m.show()
     
     
def forecast():
    '''ARIMA model to forecast
    '''
    
    from sklearn.metrics import mean_squared_error
    from pandas.tools.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_pacf
    from statsmodels.tsa.arima_model import ARIMA
    
    
    
    valuesRev=pd.read_csv("/home/deola/Documents/Opti-Num/20121001_onnetRevMatrix2.csv", names=["time","Rev"], index_col=0)
    valuesRev.index = pd.to_datetime(valuesRev.index)
    
    autocorrelation_plot(valuesRev)
    plt.savefig("/home/deola/Documents/Opti-Num/result/Autocorrelation.eps", format="eps", dpi=300)
    
    plot_pacf(valuesRev)
    plt.savefig("/home/deola/Documents/Opti-Num/result/Partial_Autocorrelation.eps", format="eps", dpi=300)
    
    model = ARIMA(valuesRev, order=(2,0,2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.savefig("/home/deola/Documents/Opti-Num/result/residual.eps", format="eps", dpi=300)
    residuals.plot(kind='kde')
    plt.savefig("/home/deola/Documents/Opti-Num/result/Gua.eps", format="eps", dpi=300)
    print(residuals.describe())
    
    forecast = model_fit.forecast(steps=24)[0]
    
    predicted= [abs(pred) for pred in forecast]
    
    Hours= [str(i)+"Hr" for i in range(24)]
    
    pred=pd.DataFrame(Hours, columns=["Time"])
    
    pred["Predicted_Revenue"]=pd.DataFrame(predicted)
    
    pred.to_csv("/home/deola/Documents/Opti-Num/result/Predict.csv", index=False)
    
    
##############################################################################################                 
##############################################################################################
##############################################################################################
if __name__=="__main__":
    
    #grouping()
    #plot_pie_chart()
    #reading_rem()
    #data_visual()
    forecast()
