# Stock Dashboard with Chatbot

An interactive stock market dashboard with a natural language chatbot interface. Users can query information about companies and their stock performance using natural language.

## Features

- Natural language query processing
- Real-time stock price visualization
- Company performance analysis
- Competitor comparison
- Market sentiment analysis
- Major shareholders information

## Example Queries

- "How is Nvidia doing?"
- "What are other companies in the same field as Nvidia?"
- "Show me Apple's performance"

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:8050`

## Gemini API Setup
1. Get your API key from https://aistudio.google.com/app/apikey
2. Create a .env file and add:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```
3. The chatbot will now understand natural language queries!

## Dependencies

- dash
- yfinance
- plotly
- pandas
- nltk
- textblob
- python-dotenv

## Note

Make sure you have a stable internet connection as the application fetches real-time stock data.
