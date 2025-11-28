GOOGLE_CREDENTIALS_FILE = "credentials.json"
SPREADSHEET_NAME = "Crypto Scanner Results"
WORKSHEET_NAME = "Scanner_Results"

"""
Crypto Scanner with Google Sheets Integration
Pushes results directly to Google Sheets for web dashboard display
"""

import time
from typing import List, Dict, Optional
from datetime import datetime
import ccxt
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ========= CONFIG =========
EXCHANGE_ID = "mexc"
QUOTE = "USDT"
TIMEFRAMES = ["1h", "4h", "1d"]
CANDLES = 150
MAX_SYMBOLS = 200
SLEEP_BETWEEN_CALLS = 0.2
MIN_VOLUME_USDT = 100000
MIN_SCORE = 60

# Google Sheets Configuration
GOOGLE_CREDENTIALS_FILE = "credentials.json"  # Download from Google Cloud Console
SPREADSHEET_NAME = "Crypto Scanner Results"   # Your Google Sheet name
WORKSHEET_NAME = "Scanner_Results"

# ==========================


def get_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def get_google_sheet():
    """
    Connect to Google Sheets
    
    Setup instructions:
    1. Go to https://console.cloud.google.com/
    2. Create a new project
    3. Enable Google Sheets API
    4. Create Service Account credentials
    5. Download JSON key file as 'credentials.json'
    6. Share your Google Sheet with the service account email
    """
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_FILE,
        scopes=scopes
    )
    
    client = gspread.authorize(creds)
    
    # Open or create spreadsheet
    try:
        spreadsheet = client.open(SPREADSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        spreadsheet = client.create(SPREADSHEET_NAME)
        spreadsheet.share('', perm_type='anyone', role='reader')  # Make readable by anyone
    
    # Get or create worksheet
    try:
        worksheet = spreadsheet.worksheet(WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=WORKSHEET_NAME,
            rows=1000,
            cols=20
        )
    
    return worksheet


def get_usdt_symbols(exchange: ccxt.Exchange, quote: str = "USDT", limit: int = 200) -> List[str]:
    markets = exchange.load_markets()
    symbols: List[str] = []
    
    try:
        tickers = exchange.fetch_tickers()
    except:
        tickers = {}
    
    for m in markets.values():
        if m.get("spot") and isinstance(m.get("symbol"), str) and m["symbol"].endswith(f"/{quote}"):
            symbol = m["symbol"]
            if symbol in tickers:
                ticker = tickers[symbol]
                volume_usdt = ticker.get("quoteVolume", 0) or 0
                if volume_usdt >= MIN_VOLUME_USDT:
                    symbols.append(symbol)
            else:
                symbols.append(symbol)
    
    symbols = sorted(set(symbols))
    if limit is not None:
        symbols = symbols[:limit]
    return symbols


def fetch_ohlcv_df(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned")
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ========== INDICATORS ==========

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def stoch_rsi_k(series: pd.Series, length: int = 14, k_smooth: int = 3) -> pd.Series:
    rsi_vals = rsi(series, length=length)
    rsi_min = rsi_vals.rolling(window=length, min_periods=length).min()
    rsi_max = rsi_vals.rolling(window=length, min_periods=length).max()
    stoch = (rsi_vals - rsi_min) / (rsi_max - rsi_min)
    stoch = stoch * 100
    k = stoch.rolling(window=k_smooth, min_periods=k_smooth).mean()
    return k


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return hist


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    
    df["rsi"] = rsi(close, length=14)
    df["stoch_rsi_k"] = stoch_rsi_k(close, length=14, k_smooth=3)
    df["macd_hist"] = macd_hist(close, fast=12, slow=26, signal=9)
    df["ema20"] = ema(close, span=20)
    df["ema50"] = ema(close, span=50)
    df["atr"] = atr(df, length=14)
    
    df["pct_change_12"] = (close / close.shift(12) - 1) * 100
    df["volume_ma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]
    
    return df


# ========== SCORING ==========

def score_bullish_setup(row_now: pd.Series, row_prev: pd.Series) -> Dict:
    score = 0
    components = {}
    
    if row_now["stoch_rsi_k"] < 20:
        stoch_score = 30
    elif row_now["stoch_rsi_k"] < 30:
        stoch_score = 20
    elif row_now["stoch_rsi_k"] < 40:
        stoch_score = 10
    else:
        stoch_score = 0
    score += stoch_score
    components["stoch_rsi"] = stoch_score
    
    if row_now["rsi"] < 30:
        rsi_score = 25
    elif row_now["rsi"] < 40:
        rsi_score = 15
    elif row_now["rsi"] < 50:
        rsi_score = 5
    else:
        rsi_score = 0
    score += rsi_score
    components["rsi"] = rsi_score
    
    if row_now["macd_hist"] < 0:
        macd_score = 15
    elif row_now["macd_hist"] < 0.5:
        macd_score = 10
    else:
        macd_score = 0
    score += macd_score
    components["macd_state"] = macd_score
    
    pct = row_now["pct_change_12"]
    if -20 <= pct <= -10:
        price_score = 20
    elif -25 <= pct < -10 or -10 < pct <= -5:
        price_score = 12
    elif -30 <= pct < -25:
        price_score = 8
    else:
        price_score = 0
    score += price_score
    components["price_pullback"] = price_score
    
    below_emas = 0
    if row_now["close"] < row_now["ema20"]:
        below_emas += 5
    if row_now["close"] < row_now["ema50"]:
        below_emas += 5
    score += below_emas
    components["ema_position"] = below_emas
    
    vol_bonus = 0
    if row_now["volume_ratio"] > 1.2:
        vol_bonus = 10
    elif row_now["volume_ratio"] > 1.0:
        vol_bonus = 5
    components["volume_bonus"] = vol_bonus
    
    return {
        "score": min(score + vol_bonus, 100),
        "base_score": score,
        "components": components
    }


def score_bearish_setup(row_now: pd.Series, row_prev: pd.Series) -> Dict:
    score = 0
    components = {}
    
    if row_now["stoch_rsi_k"] > 80:
        stoch_score = 30
    elif row_now["stoch_rsi_k"] > 70:
        stoch_score = 20
    elif row_now["stoch_rsi_k"] > 60:
        stoch_score = 10
    else:
        stoch_score = 0
    score += stoch_score
    components["stoch_rsi"] = stoch_score
    
    if row_now["rsi"] > 70:
        rsi_score = 25
    elif row_now["rsi"] > 60:
        rsi_score = 15
    elif row_now["rsi"] > 50:
        rsi_score = 5
    else:
        rsi_score = 0
    score += rsi_score
    components["rsi"] = rsi_score
    
    if row_now["macd_hist"] > 0:
        macd_score = 15
    elif row_now["macd_hist"] > -0.5:
        macd_score = 10
    else:
        macd_score = 0
    score += macd_score
    components["macd_state"] = macd_score
    
    pct = row_now["pct_change_12"]
    if 10 <= pct <= 20:
        price_score = 20
    elif 5 <= pct < 10 or 20 < pct <= 25:
        price_score = 12
    elif 25 < pct <= 30:
        price_score = 8
    else:
        price_score = 0
    score += price_score
    components["price_extension"] = price_score
    
    above_emas = 0
    if row_now["close"] > row_now["ema20"]:
        above_emas += 5
    if row_now["close"] > row_now["ema50"]:
        above_emas += 5
    score += above_emas
    components["ema_position"] = above_emas
    
    vol_bonus = 0
    if row_now["volume_ratio"] > 1.2:
        vol_bonus = 10
    elif row_now["volume_ratio"] > 1.0:
        vol_bonus = 5
    components["volume_bonus"] = vol_bonus
    
    return {
        "score": min(score + vol_bonus, 100),
        "base_score": score,
        "components": components
    }


def analyze_symbol_timeframe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    candles: int,
    min_score: int = 60,
) -> Optional[Dict]:
    try:
        df = fetch_ohlcv_df(exchange, symbol, timeframe, candles)
        df = add_indicators(df)
        df = df.dropna()
        if len(df) < 2:
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        bullish_result = score_bullish_setup(last, prev)
        bearish_result = score_bearish_setup(last, prev)
        
        if bullish_result["score"] < min_score and bearish_result["score"] < min_score:
            return None
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "close": float(last["close"]),
            "rsi": float(last["rsi"]),
            "stoch_rsi_k": float(last["stoch_rsi_k"]),
            "macd_hist": float(last["macd_hist"]),
            "pct_change_12": float(last["pct_change_12"]),
            "volume_ratio": float(last["volume_ratio"]),
            "atr_pct": float(last["atr"] / last["close"] * 100),
        }
        
        if bullish_result["score"] >= min_score:
            result["signal"] = "BULLISH"
            result["score"] = bullish_result["score"]
        
        if bearish_result["score"] >= min_score:
            if "signal" in result:
                if bearish_result["score"] > bullish_result["score"]:
                    result["signal"] = "BEARISH"
                    result["score"] = bearish_result["score"]
            else:
                result["signal"] = "BEARISH"
                result["score"] = bearish_result["score"]
        
        return result
        
    except Exception as e:
        return None


def push_to_google_sheets(worksheet, signals: List[Dict]):
    """Push all signals to Google Sheets"""
    
    # Clear existing data (keep headers)
    try:
        worksheet.clear()
    except:
        pass
    
    # Set headers
    headers = [
        'Timestamp', 'Symbol', 'Timeframe', 'Signal', 'Score',
        'Close', 'RSI', 'StochRSI', 'MACD_Hist', 'PctChange12',
        'VolumeRatio', 'ATR_Pct'
    ]
    
    # Prepare data rows
    rows = [headers]
    for sig in signals:
        rows.append([
            sig['timestamp'],
            sig['symbol'],
            sig['timeframe'],
            sig['signal'],
            sig['score'],
            sig['close'],
            sig['rsi'],
            sig['stoch_rsi_k'],
            sig['macd_hist'],
            sig['pct_change_12'],
            sig['volume_ratio'],
            sig['atr_pct']
        ])
    
    # Update sheet
    worksheet.update('A1', rows)
    
    # Format headers
    worksheet.format('A1:L1', {
        'textFormat': {'bold': True},
        'backgroundColor': {'red': 0.4, 'green': 0.5, 'blue': 0.9}
    })
    
    print(f"✅ Pushed {len(signals)} signals to Google Sheets")


def main():
    print(f"🔍 Connecting to {EXCHANGE_ID}…")
    exchange = get_exchange()
    
    print("📊 Connecting to Google Sheets...")
    worksheet = get_google_sheet()
    
    symbols = get_usdt_symbols(exchange, QUOTE, MAX_SYMBOLS)
    print(f"📈 Scanning {len(symbols)} {QUOTE} pairs\n")
    
    all_signals = []
    
    for timeframe in TIMEFRAMES:
        print(f"⏱️  Scanning {timeframe} timeframe…")
        for i, symbol in enumerate(symbols, start=1):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(symbols)}")
            
            result = analyze_symbol_timeframe(exchange, symbol, timeframe, CANDLES, MIN_SCORE)
            
            if result:
                all_signals.append(result)
                emoji = "🟢" if result["signal"] == "BULLISH" else "🔴"
                print(f"  {emoji} {symbol} [{timeframe}] Score: {result['score']:.0f}")
            
            time.sleep(SLEEP_BETWEEN_CALLS)
        
        print()
    
    # Sort by score
    all_signals.sort(key=lambda x: x["score"], reverse=True)
    
    # Push to Google Sheets
    if all_signals:
        push_to_google_sheets(worksheet, all_signals)
        print(f"\n✅ Total signals found: {len(all_signals)}")
        print(f"📊 View dashboard at: https://script.google.com/...")
    else:
        print("\n❌ No signals found")


if __name__ == "__main__":
    main()