import string
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import nltk
from textblob import TextBlob
import os
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import google.generativeai as genai
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Configure yfinance session with retries and backoff
# session = requests.Session()
# retry = Retry(
#     total=3,
#     backoff_factor=1,
#     status_forcelist=[429, 500, 502, 503, 504]
# )
# adapter = HTTPAdapter(max_retries=retry)
# session.mount('http://', adapter)
# session.mount('https://', adapter)

# Persistent session with headers to avoid bot detection
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
})

# # Configure yfinance to use our session
yf.base.session = session

def get_stock_data(symbol, period='1mo'):
    """Fetch stock data using yfinance"""
    symbol = symbol.strip()
    try:
        dat = yf.Ticker(symbol)
        hist = dat.history(period=period)
        if hist.empty:
            print(f"No data found for {symbol}")
            return pd.DataFrame()
        return hist
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_company_info(symbol):
    """Get company information including major holders"""
    try:
        time.sleep(2)
        ticker = yf.Ticker(symbol)
        
        # Basic info with fallback values
        info = {
            'name': symbol,
            'sector': 'Technology',
            'industry': 'Technology',
            'major_holders': pd.DataFrame([['Institutional Holders', 'Data Unavailable'],
                                        ['Individual Holders', 'Data Unavailable']])
        }
        
        try:
            stock_info = ticker.info
            if stock_info:
                info.update({
                    'name': stock_info.get('longName', symbol),
                    'sector': stock_info.get('sector', 'Technology'),
                    'industry': stock_info.get('industry', 'Technology')
                })
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
        
        try:
            holders = ticker.major_holders
            if not holders.empty:
                info['major_holders'] = holders
        except Exception as e:
            print(f"Error fetching holders for {symbol}: {str(e)}")
        
        return info
    except Exception as e:
        print(f"Error in get_company_info for {symbol}: {str(e)}")
        return {
            'name': symbol,
            'sector': 'Technology',
            'industry': 'Technology',
            'major_holders': pd.DataFrame([['Institutional Holders', 'Data Unavailable'],
                                        ['Individual Holders', 'Data Unavailable']])
        }

def analyze_sentiment(symbol):
    """Analyze market sentiment using recent news"""
    try:
        ticker = yf.Ticker(symbol)
        
        try:
            news = ticker.news[:5] if hasattr(ticker, 'news') and ticker.news else []
        except:
            news = []
        
        if not news:
            return "Neutral (No recent news available)"
        
        sentiment_scores = []
        for item in news:
            try:
                analysis = TextBlob(item.get('title', ''))
                sentiment_scores.append(analysis.sentiment.polarity)
            except:
                continue
        
        if not sentiment_scores:
            return "Neutral (Unable to analyze sentiment)"
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        if avg_sentiment > 0.2:
            return "Positive"
        elif avg_sentiment < -0.2:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {str(e)}")
        return "Neutral (Error in analysis)"

def create_stock_graph(symbol, period='1y'):
    """Create a stock price graph"""
    df = get_stock_data(symbol, period)
    
    if df.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this stock",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title=f'{symbol} Stock Price (No Data Available)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        name=symbol,
        line=dict(color='#00b0f0')
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white'
    )
    return fig

def create_comparison_graph(symbols, period='1y'):
    """Create a comparison graph for multiple stocks"""
    fig = go.Figure()
    
    valid_data = False
    for symbol in symbols:
        df = get_stock_data(symbol, period)
        if not df.empty:
            valid_data = True
            first_price = df['Close'].iloc[0]
            normalized_prices = (df['Close'] - first_price) / first_price * 100
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized_prices,
                name=symbol,
            ))
    
    if not valid_data:
        fig.add_annotation(
            text="No data available for comparison",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
    
    fig.update_layout(
        title='Stock Price Comparison (% Change)',
        xaxis_title='Date',
        yaxis_title='Price Change (%)',
        template='plotly_white'
    )
    return fig

def process_query(query):
    """Process natural language query using Gemini AI"""
    try:
        prompt = f"""Analyze this stock market query and return JSON with:
        1. 'company' (stock ticker symbol)
        2. 'intent' (performance, competitors, news, or unknown)
        
        Query: {query}
        
        Example response for 'How is Nvidia doing?':
        {{"company": "NVDA", "intent": "performance"}}
        """
        
        response = gemini_model.generate_content(prompt)
        result = json.loads(response.text.replace('```json', '').replace('```', '').strip())
        return result.get('intent', 'unknown'), result.get('company', '')
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return 'unknown', ''

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Company competitors mapping
COMPANY_COMPETITORS = {
    'NVDA': ['AMD', 'INTC', 'TSM', 'MU'],  # NVIDIA competitors
    'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'SMSN.IL'],  # Apple competitors
    # Add more companies and their competitors as needed
}

# Layout
app.layout = html.Div([
    dbc.Container([
        html.H1("Dashboard", className="my-4"),
        dbc.Card([
            dbc.CardBody([
                dbc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="Ask about a company (e.g., 'How is Nvidia doing?')",
                    className="mb-3"
                ),
                dbc.Button("Send", id="send-button", color="primary", className="mb-3"),
                html.Div(id="chat-output"),
                dcc.Graph(id="stock-graph"),
                html.Div(id="company-info")
            ])
        ])
    ])
])

@app.callback(
    [Output("chat-output", "children"),
     Output("stock-graph", "figure"),
     Output("company-info", "children")],
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def update_output(n_clicks, query):
    if not query:
        return "Please enter a query.", {}, ""
    
    #query_type, company = process_query(query)
    #print(query_type, company)
    query_type, company = 'performance', 'NVDA'
    
    if query_type == 'performance':
        # Get stock performance
        fig = create_stock_graph(company)
        
        # Get company info
        info = get_company_info(company)
        sentiment = analyze_sentiment(company)
        
        info_div = html.Div([
            html.H4(info['name']),
            html.P(f"Sector: {info['sector']}"),
            html.P(f"Industry: {info['industry']}"),
            html.P(f"Market Sentiment: {sentiment}"),
            html.H5("Major Holders:"),
            dbc.Table.from_dataframe(info['major_holders'])
        ])
        
        return f"Here's the analysis for {info['name']}", fig, info_div
    
    elif query_type == 'competitors':
        competitors = COMPANY_COMPETITORS.get(company, [])
        if competitors:
            fig = create_comparison_graph([company] + competitors)
            return f"Comparing {company} with its competitors", fig, ""
        else:
            return "Sorry, I don't have competitor information for this company.", {}, ""
    
    return "I'm not sure how to help with that query.", {}, ""




# Example usage
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050, ssl_context=None)
