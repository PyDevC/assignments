import plotly.express as px
import pandas as pd

def line_graph(df,title, region, product, metric, year):
    """update the line graph according to sales over a period of time

    Args:---------------------------------------------------------------------
    arguments are the filters for whole data this allows us to get data for a 
    specific event or metric.

    region: region in which sales were conducted (from Region column)
    product: it is the product name (from product column)
    metric: time period for which you need the dashboard for
    year: this is the year for which we need sales (from 'Invoice Date' column)

    Return:-------------------------------------------------------------------
    returns the fig obj from plotly.express
    """
    fig = px.line(df, x='', y='Total Sales', title=title)
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Sales')
    return fig

def bar_graph(df,title, region, year, product=None):
    """update the line graph according to sales over a period of time

    Args:---------------------------------------------------------------------
    arguments are the filters for whole data this allows us to get data for a 
    specific event or metric.

    region: region in which sales were conducted (from Region column)
    product: it is the product name (from product column)
    metric: time period for which you need the dashboard for
    year: this is the year for which we need sales (from 'Invoice Date' column)

    Return:-------------------------------------------------------------------
    returns the fig obj from plotly.express
    """
    fig = px.bar(df , x='Region', y='Total Sales', title='Sales by Region')
    fig.update_layout(xaxis_title='Region', yaxis_title='Total Sales')
    return fig

def pie_chart(df,title, region, product, year):
    """update the line graph according to sales over a period of time

    Args:---------------------------------------------------------------------
    arguments are the filters for whole data this allows us to get data for a 
    specific event or metric.

    region: region in which sales were conducted (from Region column)
    product: it is the product name (from product column)
    metric: time period for which you need the dashboard for
    year: this is the year for which we need sales (from 'Invoice Date' column)

    Return:-------------------------------------------------------------------
    returns the fig obj from plotly.express
    """
    fig = px.pie(sales_by_category, names='Product', values='Total Sales', title='Sales by Product Category')
    return fig

