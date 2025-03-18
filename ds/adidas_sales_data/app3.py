import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import datapreprocess.formatting
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import datapreprocess

df = datapreprocess.formatting.load_data()
df = datapreprocess.formatting.format(df)
df = datapreprocess.formatting.setting_dtypes(df)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    # Header
    html.Div(className='app-header', children=[
        html.H1("Sales Performance Dashboard", className='display-4')
    ]),
    html.Hr(),
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label('Select Metric:'),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Monthly Sales', 'value': 'Monthly'},
                    {'label': 'Quarterly Sales', 'value': 'Quarterly'},
                    {'label': 'Yearly Sales', 'value': 'Yearly'}
                ],
                value='Monthly',
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label('Select Region:'),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in df['Region'].unique()] + [{'label': 'All Regions', 'value': 'All'}],
                value='All',
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label('Select Product:'),
            dcc.Dropdown(
                id='product-dropdown',
                options=[{'label': category, 'value': category} for category in df['Product'].unique()] + [{'label': 'All Categories', 'value': 'All'}],
                value='All',
                clearable=False
            )
        ], width=3),
        dbc.Col([
            html.Label('Select Year:'),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': str(year)} for year in df['Invoice Date'].dt.year.unique()] + [{'label': 'All Years', 'value': 'All'}],
                value='All',
                clearable=False
            )
        ], width=3)
    ], className='mb-4'),
    # Graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id='sales-over-time'), width=8),
        dbc.Col(dcc.Graph(id='sales-by-region'), width=4)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='product-category-sales'), width=6),
        dbc.Col(dcc.Graph(id='customer-retention'), width=6)
    ]),
    # Data Insights
    dbc.Row([
        dbc.Col(html.Div(id='data-insights'), width=12)
    ])
], fluid=True)

@app.callback(
    Output('sales-over-time', 'figure'),
    [
        Input('metric-dropdown', 'value'),
        Input('region-dropdown', 'value'),
        Input('product-dropdown', 'value'),
        Input('year-dropdown', 'value')
    ]
)
def update_sales_over_time(metric, region, product, year):
    dff = df.copy()

    # Apply filters
    if region != 'All':
        dff = dff[dff['Region'] == region]
    if product != 'All':
        dff = dff[dff['Product'] == product]
    if year != 'All':
        dff = dff[dff['Invoice Date'].dt.year == int(year)]

    # Aggregate data
    if metric == 'Monthly':
        title = 'Monthly Sales'
    elif metric == 'Yearly':
        title = 'Yearly Sales'

    # Create figure
    """
    ValueError: Value of 'x' is not the name of a column in 'data_frame'. Expected one of ['Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City', 'Product', 'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method', 'Month', 'Weekday', 'Year_Month'] but received: Year
    """
    fig = px.bar(dff, x='Invoice Date', y='Operating Profit', title='Sales over time')
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Sales')
    return fig

# Callback to update sales by region graph
@app.callback(
    Output('sales-by-region', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('product-dropdown', 'value'),
        Input('year-dropdown', 'value')
    ]
)
def update_sales_by_region(region, product, year):
    dff = df.copy()

    # Apply filters
    if product != 'All':
        dff = dff[dff['Product'] == product]
    if year != 'All':
        dff = dff[dff['Invoice Date'].dt.year == int(year)]

    # Group by Region
    sales_by_region = dff.groupby('Region')['Total Sales'].sum().reset_index()

    # Create figure
    fig = px.bar(sales_by_region, x='Region', y='Total Sales', title='Sales by Region')
    fig.update_layout(xaxis_title='Region', yaxis_title='Total Sales')
    return fig

# Callback to update product category sales graph
@app.callback(
    Output('product-category-sales', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('product-dropdown', 'value'),
        Input('year-dropdown', 'value')
    ]
)
def update_product_category_sales(region, product, year):
    dff = df.copy()

    # Apply filters
    if region != 'All':
        dff = dff[dff['Region'] == region]
    if year != 'All':
        dff = dff[dff['Invoice Date'].dt.year == int(year)]

    # Group by Product
    sales_by_category = dff.groupby('Product')['Total Sales'].sum().reset_index()

    # Create figure
    fig = px.pie(sales_by_category, names='Product', values='Total Sales', title='Sales by Product Category')
    return fig

# Callback to update customer retention graph
@app.callback(
    Output('customer-retention', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('year-dropdown', 'value')
    ]
)
def update_customer_retention(region, year):
    dff = df.copy()
    
    # Apply filters
    if region != 'All':
        dff = dff[dff['Region'] == region]
    if year != 'All':
        dff = dff[dff['Invoice Date'].dt.year == int(year)]
    
    # Calculate retention
    dff['Year'] = dff['Invoice Date'].dt.year
    customers_per_year = dff.groupby('Year_Month')['Retailer'].nunique()
    retention_rate = customers_per_year.pct_change() * 100
    
    # Prepare data for plotting
    retention_df = pd.DataFrame({
        'Year': customers_per_year.index.astype(str),
        'Retention Rate (%)': retention_rate.values
    }).dropna()
    
    # Create figure
    fig = px.bar(retention_df, x='Year', y='Retention Rate (%)', title='Customer Retention Rate')
    fig.update_layout(yaxis_title='Retention Rate (%)')
    return fig

# Callback to display data insights
@app.callback(
    Output('data-insights', 'children'),
    [
        Input('region-dropdown', 'value'),
        Input('product-dropdown', 'value'),
        Input('year-dropdown', 'value')
    ]
)
def update_data_insights(region, product, year):
    dff = df.copy()

    # Apply filters
    if region != 'All':
        dff = dff[dff['Region'] == region]
    if product != 'All':
        dff = dff[dff['Product'] == product]
    if year != 'All':
        dff = dff[dff['Invoice Date'].dt.year == int(year)]

    total_sales = dff['Total Sales'].sum()
    avg_sales = dff['Total Sales'].mean()
    total_transactions = len(dff)

    insights = [
        html.H4("Data Insights"),
        html.P(f"Total Sales: ${total_sales:,.2f}"),
        html.P(f"Average Sale Amount: ${avg_sales:,.2f}"),
        html.P(f"Total Transactions: {total_transactions}")
    ]
    return insights

if __name__ == '__main__':
    app.run_server(debug=True)
