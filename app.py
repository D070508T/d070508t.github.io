import os
import time
from flask import Flask, flash, redirect, render_template, request, session, jsonify
import numpy as np
import pandas as pd
import yfinance as yf

# Configure pandas display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

app = Flask(__name__)

# Stock analysis functions
frequencies = {
    '2m': '2min',
    '15m': '15min',
    '1h': '1h',
    '90m': '90min',
    '1d': '1D',
    '5d': '5D',
    '1wk': '1W',
    '1mo': '1ME',
    '3mo': '3ME'
}

def get_stock_data(ticker, period, interval):
    try:
        print(f"Attempting to download data for {ticker} (period={period}, interval={interval})")
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        print(f"Downloaded data shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker} with period={period}, interval={interval}")
            
        df.index = pd.date_range(start=df.index[0], periods=len(df), freq=frequencies.get(interval, '1D'))
        return df
    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        raise ValueError(f"Failed to download stock data: {str(e)}")

def compute_moving_averages(df, window=25):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def relative_strength_index(df, strength=7):
    df['Price_Change'] = df['Close'].diff()
    df['Gain'] = df['Price_Change'].where(df['Price_Change'] > 0, 0)
    df['Loss'] = -df['Price_Change'].where(df['Price_Change'] < 0, 0)
    df['Avg_Gain'] = df['Gain'].ewm(alpha=1/strength, adjust=False).mean()
    df['Avg_Loss'] = df['Loss'].ewm(alpha=1/strength, adjust=False).mean()
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    return df

def rsi_signals(df, low=36, high=68, lookback_period=5):
    df['RSI_Buy_Signal'] = (df['RSI'].shift(1) < low) & (df['RSI'] >= low)
    df['RSI_Sell_Signal'] = (df['RSI'].shift(1) > high) & (df['RSI'] <= high)
    df['Recent_RSI_Buy'] = df['RSI_Buy_Signal'].rolling(lookback_period).max()
    df['Recent_RSI_Sell'] = df['RSI_Sell_Signal'].rolling(lookback_period).max()
    return df

def detect_ma_crossovers(df, ticker, lookback_period=5):
    # Create the 'Previous_Close' column
    df['Previous_Close'] = df['Close'].shift(1).fillna(0)
    
    df['Cross_Above'] = ((df['Previous_Close'] < df['EMA']) & (df[('Close', ticker)] >= df['EMA']))
    
    df['Cross_Below'] = ((df['Previous_Close'] > df['EMA']) & (df[('Close', ticker)] <= df[('EMA', '')]))
    
    # Check if a crossover happened in the last `lookback_period` days
    df['Recent_Cross_Above'] = df['Cross_Above'].rolling(lookback_period).max().astype(bool)
    
    df['Recent_Cross_Below'] = df['Cross_Below'].rolling(lookback_period).max().astype(bool)
    

    return df

def generate_signals(df):
    try:
        # Ensure all columns are properly aligned
        aligned = df[['Cross_Above', 'Recent_RSI_Buy', 'Cross_Below', 'Recent_RSI_Sell']].copy()
        
        # Generate signals with aligned data
        aligned['Full_Buy'] = aligned['Cross_Above'] & aligned['Recent_RSI_Buy']
        aligned['Full_Sell'] = aligned['Cross_Below'] & aligned['Recent_RSI_Sell']
        aligned['Final_Buy'] = aligned['Full_Buy'] & (~aligned['Full_Sell'])
        aligned['Final_Sell'] = aligned['Full_Sell'] & (~aligned['Full_Buy'])
        
        # Copy results back to original dataframe
        df[['Full_Buy', 'Full_Sell', 'Final_Buy', 'Final_Sell']] = aligned[['Full_Buy', 'Full_Sell', 'Final_Buy', 'Final_Sell']]
        return df
    except KeyError as e:
        raise ValueError(f"Missing required column: {str(e)}")
    except TypeError as e:
        raise ValueError(f"Data type mismatch: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error generating signals: {str(e)}")

def graph(period, company_ticker, ma_strength, rsi_strength, rsi_low, rsi_high, lookback):
    ticker = company_ticker.upper()

    if period == '1d':
        df = get_stock_data(ticker, '1d', '2m')
    elif period == '5d':
        df = get_stock_data(ticker, '5d', '5m')
    elif period == '1mo':
        df = get_stock_data(ticker, '1mo', '30m')
    elif period == '3mo':
        df = get_stock_data(ticker, '3mo', '60m')
    elif period == '1y':
        df = get_stock_data(ticker, '1y', '1d')
    elif period == '2y':
        df = get_stock_data(ticker, '2y', '1d')
    elif period == '5y':
        df = get_stock_data(ticker, '5y', '1wk')
    else:  # assume 10y
        df = get_stock_data(ticker, '10y', '1wk')

    length = len(df)

    df = compute_moving_averages(df, ma_strength)
    df = relative_strength_index(df, rsi_strength)
    df = rsi_signals(df, rsi_low, rsi_high, lookback)
    df = detect_ma_crossovers(df, company_ticker, lookback)
    df = generate_signals(df)
    df = df.iloc[:length]

    dates = df.index.to_numpy(dtype='datetime64[s]')
    opens = df['Open'].to_numpy(dtype=np.float64).flatten()
    highs = df['High'].to_numpy(dtype=np.float64).flatten()
    lows = df['Low'].to_numpy(dtype=np.float64).flatten()
    closes = df['Close'].to_numpy(dtype=np.float64).flatten()
    ema = df['EMA'].to_numpy(dtype=np.float64).flatten()
    
    close_text = ["Close: $" + f"{closes[i]:.2f}" for i in range(len(closes))]
    ema_text = ["EMA: $" + f"{ema[i]:.2f}" for i in range(len(ema))]
    
    # Create candlestick chart with verified data
    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        increasing_line_color='#2ECC71',  # Green
        decreasing_line_color='#E74C3C',  # Red
        increasing_fillcolor='#2ECC71',
        decreasing_fillcolor='#E74C3C',
        name="Up Trend",
        customdata=df["Close"],
        text=close_text,
        hoverinfo="text"
    )])

    # Dynamic axis scaling
    price_range = df['High'].max() - df['Low'].min()
    padding = price_range * 0.05
    
    fig.update_layout(
        title=ticker+' Stock: Last '+period,
        yaxis_title='Price ($)',
        xaxis_title='Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis_range=[df['Low'].min()-padding, df['High'].max()+padding],
        xaxis_rangeslider_visible=True,

        # Main x-axis (x1) configuration
        xaxis=dict(
            domain=[0, 1],  # Takes full width
            showspikes=True,
            title='Date'
        ),
        
        # Range slider configuration (x2)
        xaxis2=dict(
            domain=[0, 1],  # Position below main chart
            rangeslider=dict(visible=True),
            matches='x1',
            showticklabels=False,
            showgrid=False,
            showline=False,
            overlaying='x1',  # Makes it share the same space
            layer='above traces',    # Puts it behind main chart
            range=[None, None],  # Maintains auto-ranging
        )
    )

    reference_line = go.Scatter(x=dates,
                            y=ema,
                            mode="lines",
                            line=go.scatter.Line(color="gray"),
                            showlegend=True,
                            name='Exponential Moving Average',
                            customdata=df["EMA"],
                            text=ema_text,
                            hoverinfo="text"
                            )
                            

    fig.add_trace(reference_line)
    fig.update_layout(hovermode="x")
    
    if period in ['1d', '5d', '1mo', '3mo']:
        fig.update_xaxes(showticklabels=False) # hide all the xticks

    # Add clean buy/sell signal visualization
    if 'Final_Buy' in df.columns:
        buy_signals = df[df['Final_Buy']]
        y_coords = buy_signals['Close'].to_numpy(dtype=np.float64).flatten()
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=y_coords,
                mode='markers',
                marker=dict(
                    color='#00FF00',  # Bright green
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=1, color='black')
                ),
                name='BUY',
                textposition='top center',
                xaxis='x1',
                yaxis='y1',
                customdata=df["Close"],
                text='BUY',
                hoverinfo="text"
            ))
    
    if 'Final_Sell' in df.columns:
        sell_signals = df[df['Final_Sell']]
        y_coords = sell_signals['Close'].to_numpy(dtype=np.float64).flatten()
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=y_coords,
                mode='markers',
                marker=dict(
                    color='#FF0000',  # Bright red
                    size=10,
                    symbol='triangle-down',
                    line=dict(width=1, color='black')
                ),
                name='SELL',
                textposition='bottom center',
                xaxis='x1',
                yaxis='y1',
                customdata=df["Close"],
                text='SELL',
                hoverinfo="text"
            ))
    
    fig.update_layout(
        xaxis2=dict(
            showticklabels=False,
            showgrid=False,
            showline=False
        )
    )
    
    output_file = "enhanced_candlestick.html"
    fig.write_html(output_file, auto_open=True)

def create_stock_chart(df, ticker, ma_type='EMA'):
    # Prepare hover text
    close_text = ["Close: $" + f"{df['Close'][i]:.2f}" for i in range(len(df))]
    ema_text = [f"{ma_type}: $" + f"{df[ma_type][i]:.2f}" for i in range(len(df))]
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C',
        increasing_fillcolor='#2ECC71',
        decreasing_fillcolor='#E74C3C',
        customdata=df["Close"],
        text=close_text,
        hoverinfo="text"
    )])

    # Add moving average line
    reference_line = go.Scatter(
        x=df.index,
        y=df[ma_type],
        mode="lines",
        line=dict(color="gray", width=2),
        name=f'{ma_type}',
        customdata=df[ma_type],
        text=ema_text,
        hoverinfo="text"
    )
    fig.add_trace(reference_line)

    # Dynamic axis scaling
    price_range = df['High'].max() - df['Low'].min()
    padding = price_range * 0.05
    
    # Add buy/sell signals if they exist
    if 'Final_Buy' in df.columns:
        buy_signals = df[df['Final_Buy']]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                marker=dict(
                    color='#00FF00',
                    size=10,
                    symbol='triangle-up',
                    line=dict(width=1, color='black')
                ),
                name='BUY',
                hoverinfo="text",
                text='BUY'
            ))
    
    if 'Final_Sell' in df.columns:
        sell_signals = df[df['Final_Sell']]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                marker=dict(
                    color='#FF0000',
                    size=10,
                    symbol='triangle-down',
                    line=dict(width=1, color='black')
                ),
                name='SELL',
                hoverinfo="text",
                text='SELL'
            ))

    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis_range=[df['Low'].min()-padding, df['High'].max()+padding],
        xaxis_rangeslider_visible=True,
        hovermode="x",
        xaxis=dict(
            domain=[0, 1],
            showspikes=True
        ),
        xaxis2=dict(
            domain=[0, 1],
            rangeslider=dict(visible=True),
            matches='x1',
            showticklabels=False,
            showgrid=False,
            showline=False,
            overlaying='x1',
            layer='above traces'
        )
    )
    
    return fig

def create_rsi_chart(df):
    fig = go.Figure(data=[
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            line=dict(color='#9B59B6', width=2),
            name='RSI'
        )
    ])
    fig.update_layout(
        title='Relative Strength Index',
        yaxis_title='RSI',
        xaxis_title='Date',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def get_cached_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'longName': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'currentPrice': info.get('currentPrice', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A')
        }
    except:
        return {}

# Flask app setup
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname='/dash/',
    external_stylesheets=['/static/styles.css']
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24).hex())

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

def get_daily_change(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2d')
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            current = hist['Close'].iloc[-1]
            return round(((current - prev_close) / prev_close) * 100, 2)
    except:
        return None
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    # Get daily changes for sidebar stocks
    sidebar_stocks = ['AAPL', 'MSFT', 'TSLA', 'NVDA']
    stock_changes = {ticker: get_daily_change(ticker) for ticker in sidebar_stocks}
    
    if request.method == "POST":
        try:
            # Print all form submission data
            print("\n=== FORM SUBMISSION DATA ===")
            print(f"Ticker: {request.form.get('ticker', '').strip().upper()}")
            print(f"Period: {request.form.get('period', '1mo')}")
            print(f"MA Window: {request.form.get('ma_window', '25')}")
            print(f"RSI Strength: {request.form.get('rsi_strength', '7')}") 
            print(f"RSI Low: {request.form.get('rsi_low', '36')}")
            print(f"RSI High: {request.form.get('rsi_high', '68')}")
            print(f"Lookback: {request.form.get('lookback', '5')}")
            print(f"MA Type: {request.form.get('ma_type', 'EMA')}")
            print(f"Signal Priority: {request.form.get('signal_priority', 'RSI')}")
            print("===========================\n")

            # Verify form data is present
            if not all(request.form.get(field) for field in ['ticker', 'period']):
                flash("Missing required form fields")
                return redirect("/")
            
            ticker = request.form.get("ticker", "").strip().upper()
            if not ticker:
                flash("Please enter a stock ticker")
                return redirect("/")
                
            period = request.form.get("period", "1mo")
            
            # Get stock info with retry and fallback
            stock_info = {}
            retries = 0
            while retries < 3 and not stock_info.get('currentPrice'):
                try:
                    stock_info = get_cached_info(ticker)
                    if not stock_info.get('currentPrice'):
                        time.sleep(1)  # Wait before retrying
                        retries += 1
                except Exception as e:
                    print(f"Error getting stock info (attempt {retries + 1}): {str(e)}")
                    time.sleep(1)
                    retries += 1
                        
            if not stock_info.get('currentPrice'):
                print("Using fallback method to get current price")
                try:
                    hist = yf.Ticker(ticker).history(period='1d')
                    if not hist.empty:
                        stock_info['currentPrice'] = hist['Close'].iloc[-1]
                except Exception as e:
                    print(f"Fallback method failed: {str(e)}")
                    flash("Could not retrieve stock information")
                    return redirect("/")
            
            results = {
                'ticker': ticker,
                'company': stock_info.get('longName', ticker),
                'sector': stock_info.get('sector', 'N/A'),
                'current_price': stock_info.get('currentPrice', 'N/A'),
                'market_cap': stock_info.get('marketCap', 'N/A'),
                'high_52wk': stock_info.get('fiftyTwoWeekHigh', 'N/A'),
                'low_52wk': stock_info.get('fiftyTwoWeekLow', 'N/A')
            }

            form_results = {
                'ticker': request.form.get("ticker", ""),
                'period': request.form.get("period", "1 Month"),
                'ma_strength': int(request.form.get("ma_strength", 25)),
                'ma_window': int(request.form.get("ma_window", 25)),
                'rsi_strength': int(request.form.get("rsi_strength", 7)),
                'lookback': int(request.form.get("lookback", 5)),
                'rsi_low': int(request.form.get("rsi_low", 36)),
                'rsi_high': int(request.form.get("rsi_high", 68))
            }

            try:
                time.sleep(1)
                df = get_stock_data(ticker, period, '1d')
            except ValueError as e:
                flash(str(e))
                return redirect("/")

            df = compute_moving_averages(df, 25)
            df = relative_strength_index(df, 7)
            df = rsi_signals(df, 36, 68, 10)
            df = detect_ma_crossovers(df, ticker, 10)
            df = generate_signals(df)
            
            # Get news for this stock with retry logic
            formatted_news = []
            retries = 0
            while retries < 3 and not formatted_news:
                try:
                    for i in range(3):
                        content = yf.Ticker(ticker).news[i].get('content')
                        if content == {}:
                            print("ERROR: NO ARTICLE DATA")
                        else:
                            article_content = {}
                            article_content['title'] = content.get('title')
                            article_content['summary'] = content.get('summary')
                            article_content['pubDate'] = content.get('pubDate')
                            #article_content['originalUrl'] = content.get('originalUrl')
                            #article_content['provider'] = content.get('provider').get('displayName')

                            formatted_news.append(article_content)
                except Exception as e:
                    print(f"Error getting news (attempt {retries + 1}): {str(e)}")
                    time.sleep(1)
                    retries += 1
        
            # Ensure all stock data is properly populated
            current_price = stock_info.get('currentPrice', None)
            if current_price is None:
                try:
                    current_price = df['Close'].iloc[-1]
                except:
                    current_price = 'N/A'

            results['change'] = round(((df[('Close', ticker)].iloc[-1] - df[('Close', ticker)].iloc[-2]) / df[('Close', ticker)].iloc[-2]) * 100, 2) if len(df) > 1 else 'N/A'
            graph(form_results.get('period'), form_results.get('ticker'), form_results.get('ma_strength'), form_results.get('rsi_strength'), form_results.get('rsi_low'), form_results.get('rsi_high'), form_results.get('lookback')) 

            return render_template("index.html", results=results, request=request, stock_changes=stock_changes, current_price=current_price)
            
        except Exception as e:
            flash(f"Error analyzing stock: {str(e)}")
            return redirect("/")

    return render_template("index.html", results=None, stock_changes=stock_changes, stock_data=None, request=request, current_price=0)

dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@dash_app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    return html.Div([
        html.H1('Stock Analysis Dashboard'),
        dcc.Graph(id='stock-chart'),
        dcc.Graph(id='rsi-chart')
    ])

@app.route("/get_news", methods=["GET"])
def get_news():
    try:
        news = yf.Ticker("^GSPC").news
        return jsonify([{
            'title': item['title'],
            'publisher': item['publisher'],
            'link': item['link'],
            'date': item['providerPublishTime']
        } for item in news[:5]])
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    try:
        data = request.get_json()
        ticker = data['ticker'].upper()
        period = data.get('period', '1mo')
        
        df = get_stock_data(ticker, period, '1d')
        df = compute_moving_averages(df)
        df = relative_strength_index(df)
        
        return jsonify({
            'success': True,
            'data': df.reset_index().to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
