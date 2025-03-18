import pandas as pd
import numpy as np
import os

def load_data():
    df = pd.read_excel("../data/adidas-sales-data/Adidas US Sales Datasets.xlsx")
    return df

def format(df):
    df = df.drop([0,1,2]).reset_index(drop=True) #excluding first 3 rows
    df.columns = df.iloc[0]
    df = df.drop([0]).reset_index(drop=True)
    df = df.drop(df.columns[0], axis = 1)
    return df

def setting_dtypes(df):
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
    df['Price per Unit'] = pd.to_numeric(df['Price per Unit'])
    df['Units Sold'] = pd.to_numeric(df['Units Sold'])
    df['Total Sales'] = pd.to_numeric(df['Total Sales'])
    df['Operating Profit'] = pd.to_numeric(df['Operating Profit'])
    df['Operating Margin'] = pd.to_numeric(df['Operating Margin'])
    df['Month'] = df['Invoice Date'].dt.month
    df['Weekday']= df['Invoice Date'].dt.day_name()
    df['Units Sold'] = pd.to_numeric(df['Units Sold'])
    df['Retailer'] = df['Retailer'].astype(str)
    df['Year_Month'] = pd.to_datetime(df['Invoice Date']).dt.to_period('M')
    df = df.sort_values(by='Year_Month')
    return df
