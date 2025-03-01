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
    """Get company information including market mood"""
    try:
        time.sleep(2)
        ticker = yf.Ticker(symbol)
        
        # Basic info with fallback values
        info = {
            'name': symbol,
            'sector': 'Technology',
            'industry': 'Technology',
            'market_mood': {'buy': 0, 'hold': 0, 'sell': 0}
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
            # Get recommendations data
            recommendations = ticker.recommendations
            if not recommendations.empty:
                # Get the most recent recommendation (first row)
                recent_rec = recommendations.iloc[0]
                
                # Extract buy, hold, sell counts
                buy_count = int(recent_rec.get('strongBuy', 0)) + int(recent_rec.get('buy', 0))
                hold_count = int(recent_rec.get('hold', 0))
                sell_count = int(recent_rec.get('sell', 0)) + int(recent_rec.get('strongSell', 0))
                
                info['market_mood'] = {
                    'buy': buy_count,
                    'hold': hold_count,
                    'sell': sell_count
                }
        except Exception as e:
            print(f"Error fetching recommendations for {symbol}: {str(e)}")
        
        return info
    except Exception as e:
        print(f"Error in get_company_info for {symbol}: {str(e)}")
        return {
            'name': symbol,
            'sector': 'Technology',
            'industry': 'Technology',
            'market_mood': {'buy': 0, 'hold': 0, 'sell': 0}
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
        
        Example response for 'Compare Apple with its competitors':
        {{"company": "AAPL", "intent": "competitors"}}
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

def create_market_mood_chart(market_mood):
    """Create a pie chart showing market mood based on analyst recommendations"""
    # Prepare data for pie chart
    labels = ['Buy', 'Hold', 'Sell']
    values = [market_mood['buy'], market_mood['hold'], market_mood['sell']]
    
    # Using vibrant colors for maximum visibility
    colors = ['#00cc00', '#ffcc00', '#ff0000']  # Green, Yellow, Red
    
    # Calculate total recommendations for percentage calculation
    total_recs = sum(values)
    
    if total_recs == 0:
        # No data available
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Market Mood",
            template="plotly_white",
            height=350  # Increased height
        )
        return fig
    
    # Create hover text with percentages
    hover_text = [
        f"Buy: {market_mood['buy']} ({market_mood['buy']/total_recs*100:.1f}%)",
        f"Hold: {market_mood['hold']} ({market_mood['hold']/total_recs*100:.1f}%)",
        f"Sell: {market_mood['sell']} ({market_mood['sell']/total_recs*100:.1f}%)"
    ]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='percent+label',  # Added labels to the chart
        hoverinfo='text',
        hovertext=hover_text,
        textfont=dict(size=14)  # Larger text
    )])
    
    # Update layout
    fig.update_layout(
        title="Market Mood (Analyst Recommendations)",
        title_font=dict(size=16),
        template="plotly_white",
        height=350,  # Increased height
        margin=dict(t=50, b=20, l=20, r=20)  # Adjusted margins
    )
    
    return fig

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
                                html.Div(id="stock-graph-container", className="h-100", children=[
                                    html.Div(
                                        "Enter a stock query to display data",
                                        className="h-100 d-flex justify-content-center align-items-center text-muted"
                                    )
                                ], style={"height": "40vh"})
                            ], width=8),
                            
                            # Company info and market mood
                            dbc.Col([
                                html.Div(id="company-info", className="mb-3"),
                                html.Div(id="market-mood-container", className="h-100", children=[
                                    # Initially empty
                                ])
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
                        html.Div(
                            id="chat-output", 
                            className="chat-response border rounded mb-3", 
                            style={"height": "65vh", "overflow-y": "auto", "padding": "15px"},
                            children=[
                                html.P("Welcome to the Stock Dashboard! Ask me about any stock, for example:"),
                                html.Ul([
                                    html.Li("How is Apple doing?"),
                                    html.Li("Show me Tesla stock performance"),
                                    html.Li("What's the market sentiment for Microsoft?")
                                ])
                            ]
                        ),
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
     Output("stock-graph-container", "children"),
     Output("company-info", "children"),
     Output("market-mood-container", "children"),
     Output("news-widget", "children")],
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value")],
    prevent_initial_call=True
)
def update_output(n_clicks, query):
    if not query:
        return (
            html.P("Please enter a query about a stock."),
            html.Div(
                "Enter a stock query to display data",
                className="h-100 d-flex justify-content-center align-items-center text-muted"
            ),
            html.Div(""),
            html.Div(""),
            html.Div("")
        )
    
    try:
        # Process the query to extract intent and company
        intent, symbol = process_query(query)
        
        if not symbol:
            return (
                html.P(f"I couldn't identify a stock symbol in your query. Please try again with a specific company."),
                html.Div(
                    "Enter a stock query to display data",
                    className="h-100 d-flex justify-content-center align-items-center text-muted"
                ),
                html.Div(""),
                html.Div(""),
                html.Div("")
            )
        
        # Handle competitors intent differently
        if intent == "competitors":
            # Get competitors for the symbol
            competitors = COMPANY_COMPETITORS.get(symbol, [])
            
            if not competitors:
                # If no competitors are defined, return a message
                return (
                    html.P(f"I don't have competitor information for {symbol}. Try another company or a different query."),
                    html.Div(
                        f"No competitor data available for {symbol}",
                        className="h-100 d-flex justify-content-center align-items-center text-muted"
                    ),
                    html.Div(""),
                    html.Div(""),
                    html.Div("")
                )
            
            # Create comparison graph with the symbol and its competitors
            comparison_fig = create_comparison_graph([symbol] + competitors)
            
            # Create a response message
            response = html.Div([
                html.P(f"Comparing {symbol} with its competitors:"),
                html.Ul([html.Li(comp) for comp in competitors])
            ])
            
            # Return only the comparison graph, leaving other components empty
            return (
                response,
                dcc.Graph(figure=comparison_fig, style={"height": "70vh"}),  # Larger graph for comparison
                html.Div(""),  # Empty company info
                html.Div(""),  # Empty market mood
                html.Div("")   # Empty news widget
            )
        
        # Regular performance intent flow
        # Get company info
        company_info = get_company_info(symbol)
        
        # Create stock graph
        stock_fig = create_stock_graph(symbol)
        
        # Create market mood chart
        market_mood_chart = create_market_mood_chart(company_info['market_mood'])
        
        # Get news sentiment
        sentiment_label, news_items = analyze_sentiment(symbol)
        
        # Create news widget
        news_widget = create_news_widget(news_items)
        
        # Create company info component
        company_info_component = html.Div([
            html.H4(company_info['name']),
            html.P(f"Sector: {company_info['sector']}"),
            html.P(f"Industry: {company_info['industry']}"),
            html.P(f"Sentiment: {sentiment_label}", className=f"{'text-success' if 'Positive' in sentiment_label else 'text-danger' if 'Negative' in sentiment_label else 'text-warning'}")
        ])
        
        # Create response message
        response = html.Div([
            html.P(f"Analyzing {company_info['name']} ({symbol})"),
            html.P(f"I've gathered information about {company_info['name']} including its stock performance, market mood based on analyst recommendations, and recent news sentiment.")
        ])
        
        # Return actual graph and chart components
        stock_graph_component = dcc.Graph(figure=stock_fig, style={"height": "40vh"})
        market_mood_component = dcc.Graph(figure=market_mood_chart, style={"height": "35vh"})
        
        return response, stock_graph_component, company_info_component, market_mood_component, news_widget
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return (
            html.P(f"Error: {str(e)}"),
            html.Div(
                f"Error retrieving stock data: {str(e)}",
                className="h-100 d-flex justify-content-center align-items-center text-danger"
            ),
            html.Div("Error retrieving company information"),
            html.Div(""),
            html.Div("Error retrieving news")
        )

# Example usage
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050, ssl_context=None)
