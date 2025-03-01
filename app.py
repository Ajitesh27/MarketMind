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
import re

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
            news = ticker.news[:25] if hasattr(ticker, 'news') and ticker.news else []
        except Exception as e:
            print(f"Error accessing news: {str(e)}")
            news = []
        
        if not news:
            return "Neutral (No recent news available)", []
        
        news_with_sentiment = []
        sentiment_scores = []
        
        for item in news:
            try:
                # Access the nested content structure
                content = item.get('content', {})
                canonical_url = content.get('canonicalUrl', '')
                
                # Extract text and URL from the content dictionary
                title = content.get('title', '')
                description = content.get('description', '')
                summary = content.get('summary', '')
                url = canonical_url.get('url', '')
                
                # Use title first, then description, then summary
                text = title or description or summary
                
                # Clean HTML tags from text if present
                if '<' in text and '>' in text:
                    text = re.sub('<.*?>', ' ', text)
                
                if text:
                    analysis = TextBlob(text)
                    sentiment = analysis.sentiment.polarity
                    
                    # Add to our lists
                    sentiment_scores.append(sentiment)
                    news_with_sentiment.append({
                        'title': title,
                        'sentiment': sentiment,
                        'url': url
                    })
            except Exception as e:
                print(f"Error analyzing news item: {str(e)}")
                continue
        
        if not sentiment_scores:
            return "Neutral (Unable to analyze sentiment)", []
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Determine overall sentiment
        if avg_sentiment > 0.2:
            overall_sentiment = "Positive"
        elif avg_sentiment < -0.2:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
            
        return overall_sentiment, news_with_sentiment
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {str(e)}")
        return "Neutral (Error in analysis)", []

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

def create_news_widget(news_with_sentiment):
    """Create a widget displaying positive and negative news"""
    if not news_with_sentiment:
        return html.Div("No news available for this stock")
    
    # Sort news by sentiment
    sorted_news = sorted(news_with_sentiment, key=lambda x: x['sentiment'], reverse=True)
    
    # Get top 3 positive and negative news
    positive_news = [n for n in sorted_news if n['sentiment'] > 0][:3]
    negative_news = [n for n in sorted_news if n['sentiment'] < 0][:3]
    
    # Create tables for positive and negative news
    return html.Div([
        html.H4("Latest News Sentiment", className="mt-3 mb-4"),
        dbc.Row([
            # Positive news
            dbc.Col([
                html.H5("Positive News", className="text-success border-bottom pb-2"),
                html.Div([
                    html.Div([
                        html.A(
                            news['title'] or "No title available", 
                            href=news['url'],
                            target="_blank",
                            className="text-decoration-none"
                        ),
                        html.Small(f" (Sentiment: {news['sentiment']:.2f})", className="text-muted")
                    ], className="mb-2 p-2 border-bottom") 
                    for news in positive_news
                ]) if positive_news else html.P("No positive news found")
            ], width=6),
            
            # Negative news
            dbc.Col([
                html.H5("Negative News", className="text-danger border-bottom pb-2"),
                html.Div([
                    html.Div([
                        html.A(
                            news['title'] or "No title available", 
                            href=news['url'],
                            target="_blank",
                            className="text-decoration-none"
                        ),
                        html.Small(f" (Sentiment: {news['sentiment']:.2f})", className="text-muted")
                    ], className="mb-2 p-2 border-bottom") 
                    for news in negative_news
                ]) if negative_news else html.P("No negative news found")
            ], width=6)
        ])
    ], className="mt-4 p-3 border rounded")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Company competitors mapping
COMPANY_COMPETITORS = {
    'NVDA': ['AMD', 'INTC', 'TSM', 'MU'],  # NVIDIA competitors
    'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'SMSN.IL'],  # Apple competitors
    # Add more companies and their competitors as needed
}

# Layout
app.layout = html.Div(className="vh-100 d-flex justify-content-center align-items-center", style={"background-color": "#f8f9fa"}, children=[
    # Main container - 80% of screen width
    dbc.Container(className="h-100 py-4", style={"width": "80%", "max-width": "80%"}, children=[
        html.H1("Stock Dashboard", className="mb-4 text-center"),
        
        # Main row - side by side layout
        dbc.Row(className="h-90", children=[
            # Left side - Dashboard (70% of the container)
            dbc.Col(width=8, className="h-100", children=[
                dbc.Card(className="h-100 shadow", children=[
                    dbc.CardBody([
                        # Stock graph and company info side by side
                        dbc.Row([
                            # Stock graph
                            dbc.Col([
                                dcc.Graph(id="stock-graph", style={"height": "40vh"})
                            ], width=8),
                            
                            # Company info
                            dbc.Col([
                                html.Div(id="company-info", className="h-100")
                            ], width=4)
                        ]),
                        
                        # News widget
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="news-widget", className="mt-4")
                            ])
                        ])
                    ])
                ])
            ]),
            
            # Right side - Chatbot (30% of the container)
            dbc.Col(width=4, className="h-100", children=[
                dbc.Card(className="h-100 shadow", children=[
                    dbc.CardHeader("AI Stock Assistant"),
                    dbc.CardBody([
                        html.Div(id="chat-output", 
                                className="chat-response border rounded mb-3", 
                                style={"height": "65vh", "overflow-y": "auto", "padding": "15px"}),
                        dbc.Input(
                            id="chat-input",
                            type="text",
                            placeholder="Ask about a company (e.g., 'How is Nvidia doing?')",
                            className="mb-3"
                        ),
                        dbc.Button("Send", id="send-button", color="primary", className="w-100")
                    ])
                ])
            ])
        ])
    ])
])

@app.callback(
    [Output("chat-output", "children"),
     Output("stock-graph", "figure"),
     Output("company-info", "children"),
     Output("news-widget", "children")],
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def update_output(n_clicks, query):
    if not query:
        return "Please enter a query.", {}, "", ""
    
    query_type, company = process_query(query)
    print(query_type, company)
    
    if query_type == 'performance':
        # Get stock performance
        fig = create_stock_graph(company)
        
        # Get company info
        info = get_company_info(company)
        sentiment, news_with_sentiment = analyze_sentiment(company)
        
        # Create company info div
        info_div = html.Div([
            html.H3(info['name'], className="mb-4"),
            html.Div([
                html.H5("Company Information", className="border-bottom pb-2"),
                html.P(f"Sector: {info['sector']}", className="mb-1"),
                html.P(f"Industry: {info['industry']}", className="mb-1"),
                html.P(f"Market Sentiment: {sentiment}", className="mb-3"),
                
                html.H5("Major Holders", className="border-bottom pb-2 mt-4"),
                dbc.Table.from_dataframe(info['major_holders'], striped=True, bordered=True, hover=True, size="sm")
            ], className="p-3")
        ], className="h-100 overflow-auto")
        
        # Create news widget
        news_widget = create_news_widget(news_with_sentiment)
        
        return f"Here's the analysis for {info['name']}", fig, info_div, news_widget
    
    elif query_type == 'competitors':
        competitors = COMPANY_COMPETITORS.get(company, [])
        if competitors:
            fig = create_comparison_graph([company] + competitors)
            
            # Create a simple info panel for competitors
            info_div = html.Div([
                html.H3(f"{company} Comparison", className="mb-4"),
                html.H5("Comparing With:", className="border-bottom pb-2"),
                html.Ul([html.Li(comp) for comp in competitors], className="mt-3")
            ], className="p-3")
            
            # Get sentiment for the main company
            sentiment, news_with_sentiment = analyze_sentiment(company)
            news_widget = create_news_widget(news_with_sentiment)
            
            return f"Comparing {company} with its competitors", fig, info_div, news_widget
        else:
            return "Sorry, I don't have competitor information for this company.", {}, "", ""
    
    return "I'm not sure how to help with that query.", {}, "", ""

# Example usage
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050, ssl_context=None)
