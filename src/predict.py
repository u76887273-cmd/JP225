import os
import sys
import joblib
import psycopg
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ============================================================
# JAPAN225 QUANTUM SUPREME — LIVE PREDICTION ENGINE
# THE "GOD MODE" GRID: MAX 5 TRADES + 4/8/16/32h DELAYS + ATR 5.0
# ============================================================

# Database connection from GitHub Secrets
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable is missing.")
    sys.exit(1)

# Paths for GitHub Actions runner
MODEL_DIR = "models"

# ------------------------------------------------------------
# GRID MACRO SETTINGS
# ------------------------------------------------------------
MAX_OPEN_TRADES = 5
ATR_MULTIPLIER = 5.0
TIME_DELAYS_HOURS = [4, 8, 16, 32] 
FEE = 15.0 # Broker spread adjustment

# ------------------------------------------------------------
# DATABASE CONNECTION & SCHEMA VERIFICATION
# ------------------------------------------------------------
try:
    conn = psycopg.connect(DATABASE_URL)
    conn.autocommit = True 
except Exception as e:
    print(f"Database connection failed: {e}")
    sys.exit(1)

cur = conn.cursor()

# Ensure the log table exists in Postgres
cur.execute("""
CREATE TABLE IF NOT EXISTS supreme_grid_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ UNIQUE,
    price REAL,
    signal TEXT,
    reason TEXT,
    entry_zone TEXT,
    entry REAL,
    take_profit REAL,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
""")

print("Fetching recent market data...")
# Increased to 200 so moving averages (ema50, atr50) have room to calculate!
cur.execute("SELECT * FROM features_m5 ORDER BY timestamp DESC LIMIT 200;")
columns = [desc[0] for desc in cur.description]
data = cur.fetchall()

if not data:
    print("No data found in features_m5. Exiting.")
    sys.exit(0)

bars = pd.DataFrame(data, columns=columns)
bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
bars = bars.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------------------
# DYNAMIC FEATURE ENGINEERING (MATCHING COLAB)
# ------------------------------------------------------------
print("Engineering advanced features for AI...")

# Returns
bars['return_1'] = bars['close'].pct_change(1) * 100
bars['return_3'] = bars['close'].pct_change(3) * 100
bars['return_6'] = bars['close'].pct_change(6) * 100

# Candle shapes
bars['body'] = bars['close'] - bars['open']
bars['upper_wick'] = bars['high'] - bars[['open', 'close']].max(axis=1)
bars['lower_wick'] = bars[['open', 'close']].min(axis=1) - bars['low']

# EMAs
bars['ema20'] = bars['close'].ewm(span=20, adjust=False).mean()
bars['ema50'] = bars['close'].ewm(span=50, adjust=False).mean()
bars['ema_gap'] = bars['ema20'] - bars['ema50']
bars['ema20_slope'] = bars['ema20'].diff()

# ATRs
bars['prev_close'] = bars['close'].shift(1)
bars['tr1'] = bars['high'] - bars['low']
bars['tr2'] = (bars['high'] - bars['prev_close']).abs()
bars['tr3'] = (bars['low'] - bars['prev_close']).abs()
bars['tr'] = bars[['tr1', 'tr2', 'tr3']].max(axis=1)
bars['atr14'] = bars['tr'].rolling(14).mean()
bars['atr50'] = bars['tr'].rolling(50).mean()
bars['relative_atr'] = bars['atr14'] / bars['atr50']

# RSI 14
delta = bars["close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
bars["rsi14"] = 100 - (100 / (1 + rs))

# VWAP (Daily Reset)
bars['date'] = bars['timestamp'].dt.date
bars['typical_price'] = (bars['high'] + bars['low'] + bars['close']) / 3

# Safe VWAP calculation avoiding the apply() ValueError
cum_vp = (bars['typical_price'] * bars['volume']).groupby(bars['date']).cumsum()
cum_v = bars['volume'].groupby(bars['date']).cumsum()
bars['vwap'] = cum_vp / cum_v

bars['vwap_distance'] = bars['close'] - bars['vwap']

# Opening Range (OR) - Using first 1 hour (12 M5 bars) of the day
or_df = bars.groupby('date').head(12).groupby('date').agg({
    'high': 'max', 'low': 'min', 'open': 'first'
})
bars['opening_high'] = bars['date'].map(or_df['high'])
bars['opening_low'] = bars['date'].map(or_df['low'])
bars['opening_open'] = bars['date'].map(or_df['open'])

bars['dist_or_high'] = bars['close'] - bars['opening_high']
bars['dist_or_low'] = bars['close'] - bars['opening_low']
bars['dist_open_price'] = bars['close'] - bars['opening_open']
bars['inside_or'] = ((bars['close'] < bars['opening_high']) & (bars['close'] > bars['opening_low'])).astype(int)

# Sessions
hour = bars['timestamp'].dt.hour
bars['tokyo'] = ((hour >= 0) & (hour < 8)).astype(int)
bars['london'] = ((hour >= 8) & (hour < 16)).astype(int)
bars['newyork'] = ((hour >= 13) & (hour < 21)).astype(int)

# Get the most recent row after calculating all rolling features
latest = bars.iloc[-1:]

# ------------------------------------------------------------
# LOAD SUPREME MODELS & PREDICT
# ------------------------------------------------------------
print("Loading Supreme ML Models...")
FEATURES = joblib.load(f"{MODEL_DIR}/features.joblib")
upside_model = joblib.load(f"{MODEL_DIR}/upside30.joblib")
downside_model = joblib.load(f"{MODEL_DIR}/downside30.joblib")
width_model = joblib.load(f"{MODEL_DIR}/width30.joblib")

X = latest[FEATURES].fillna(0)

price = float(latest.close.iloc[0])
atr   = float(latest.atr14.iloc[0])
rsi   = float(latest.rsi14.iloc[0])
time_stamp = latest.timestamp.iloc[0]

pred_upside   = float(upside_model.predict(X)[0])
pred_downside = float(downside_model.predict(X)[0])
pred_width    = float(width_model.predict(X)[0])

# ------------------------------------------------------------
# PROJECT ALL ZONES
# ------------------------------------------------------------
proj_high = price + pred_upside
proj_low  = price - pred_downside
midline = (proj_high + proj_low) / 2
zone_height = pred_width * 0.15

buy_zone_bottom = proj_low - FEE
buy_zone_top = (proj_low + zone_height) - FEE
buy_zone_str = f"{buy_zone_bottom:.1f} to {buy_zone_top:.1f}"
buy_limit = buy_zone_top
buy_tp = midline

sell_zone_bottom = proj_high - zone_height
sell_zone_top = proj_high
sell_zone_str = f"{sell_zone_bottom:.1f} to {sell_zone_top:.1f}"
sell_limit = sell_zone_bottom
sell_tp = midline - FEE

# ------------------------------------------------------------
# RAW SIGNAL GENERATION 
# ------------------------------------------------------------
raw_signal = "WAIT"

if pred_upside > (pred_downside * 1.5) and rsi < 65:
    raw_signal = "BUY"
elif pred_downside > (pred_upside * 1.5) and rsi > 35:
    raw_signal = "SELL"

# ------------------------------------------------------------
# THE GRID DEFENSE SYSTEM
# ------------------------------------------------------------
signal = raw_signal
reason = "Autonomous Grid Execution" if raw_signal != "WAIT" else "Not enough AI conviction (WAIT)"
blocked_reason = None
elapsed_hours = 0.0

try:
    active_df = pd.read_sql("SELECT * FROM supreme_grid_log WHERE status IN ('PENDING', 'OPEN') ORDER BY timestamp", conn)
    num_open = len(active_df)
except:
    num_open = 0
    active_df = pd.DataFrame()

if raw_signal != "WAIT":
    if num_open >= MAX_OPEN_TRADES:
        signal = "WAIT"
        blocked_reason = "MAX_5_OPEN_TRADES_HIT"
        
    elif num_open > 0:
        active_df['timestamp'] = pd.to_datetime(active_df['timestamp'], utc=True)
        last_trade_time = active_df['timestamp'].iloc[-1]
        elapsed_hours = (time_stamp - last_trade_time).total_seconds() / 3600.0
        
        delay_idx = min(num_open - 1, len(TIME_DELAYS_HOURS) - 1)
        req_hours = TIME_DELAYS_HOURS[delay_idx]
        
        if elapsed_hours < req_hours:
            signal = "WAIT"
            blocked_reason = f"TIME_DELAY (Needs {req_hours}h, Elapsed {elapsed_hours:.1f}h)"
        else:
            req_dist = atr * ATR_MULTIPLIER
            too_close = False
            for _, row in active_df.iterrows():
                if abs(price - row['entry']) < req_dist:
                    too_close = True
                    break
                    
            if too_close:
                signal = "WAIT"
                blocked_reason = f"ATR_DISTANCE (Needs {req_dist:.1f} pts)"

if blocked_reason:
    reason = f"BLOCKED: {blocked_reason}"

# ------------------------------------------------------------
# ASSIGN FINAL VARIABLES FOR DATABASE
# ------------------------------------------------------------
if signal == "BUY":
    chart_zone, chart_entry, chart_tp = buy_zone_str, buy_limit, buy_tp
elif signal == "SELL":
    chart_zone, chart_entry, chart_tp = sell_zone_str, sell_limit, sell_tp
else:
    chart_zone = f"B: {buy_limit:.1f} | S: {sell_limit:.1f}"
    chart_entry = "-" 
    chart_tp = midline


# ------------------------------------------------------------
# INSERT INTO LOG (SUPABASE via psycopg3)
# ------------------------------------------------------------
ts = str(time_stamp)
entry_val = None if chart_entry == "-" else float(chart_entry)
tp_val    = None if chart_tp == "-" else float(chart_tp)
zone_val  = None if chart_zone == "-" else chart_zone

cur.execute("""
INSERT INTO supreme_grid_log (
    timestamp, price, signal, reason, entry_zone, entry, take_profit, status
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (timestamp) DO NOTHING;
""", (
    ts,
    round(price, 1),
    signal,
    reason,
    zone_val,
    entry_val,
    tp_val,
    "PENDING" if signal != "WAIT" else "IGNORED"
))

cur.close()
conn.close()

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------
print("\n" + "="*64)
print("JAPAN225 QUANTUM SUPREME — CLOUD GRID ENGINE")
print("="*64)
print(f"UTC Time : {time_stamp}")
print(f"Current  : {price:.1f}")
print(f"ATR (14) : {atr:.1f}")
print(f"RSI (14) : {rsi:.1f}")
print("-" * 64)
print(f"Active Grid Positions : {num_open} / {MAX_OPEN_TRADES}")
if num_open > 0:
    print(f"Time Since Last Trade : {elapsed_hours:.1f} Hours")
    print(f"Current Distance Req. : {atr * ATR_MULTIPLIER:.1f} pts")
print("-" * 64)
print(f"PROJECTED BUY ZONE  : {buy_zone_str} (Limit: {buy_limit if type(buy_limit) == str else round(buy_limit, 1)})")
print(f"PROJECTED SELL ZONE : {sell_zone_str} (Limit: {sell_limit if type(sell_limit) == str else round(sell_limit, 1)})")
print(f"EQUILIBRIUM TP      : {midline:.1f}")
print("-" * 64)
print(f"RAW AI SIGNAL : {raw_signal}")
print(f"FINAL SIGNAL  : {signal}")
print(f"REASON        : {reason}")
if signal != "WAIT":
    print(f">> EXECUTING LIMIT ORDER AT {chart_entry:.1f} <<")
print("="*64)
