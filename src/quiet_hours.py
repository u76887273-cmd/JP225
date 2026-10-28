import os
import sys
import psycopg
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_URL = os.environ.get("DATABASE_URL")

def load_bars():
    df = pd.DataFrame()

    # 1. Try using train.py's built-in loader first
    try:
        from train import sync_and_load_all_m5_bars
        df = sync_and_load_all_m5_bars()
        if df is not None and not df.empty:
            print(f"Loaded {len(df):,} bars via train.sync_and_load_all_m5_bars()")
    except Exception as e:
        print(f"Note on train loader: {e}")

    # 2. Auto-discover OHLC table in Supabase if needed
    if (df is None or df.empty) and DB_URL:
        try:
            with psycopg.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND column_name IN ('timestamp', 'high', 'low')
                        GROUP BY table_name
                        HAVING COUNT(DISTINCT column_name) = 3;
                    """)
                    tables = [r[0] for r in cur.fetchall()]

                    best_rows = []
                    for t in tables:
                        cur.execute(f"SELECT timestamp, high, low FROM {t} ORDER BY timestamp ASC;")
                        rows = cur.fetchall()
                        if len(rows) > len(best_rows):
                            best_rows = rows

                    if best_rows:
                        df = pd.DataFrame(best_rows, columns=["timestamp", "high", "low"])
        except Exception as e:
            print(f"Note on DB auto-discovery: {e}")

    # 3. Also check local CSV files
    for csv_path in ["data_m5.csv", "src/data_m5.csv", "jp225_m5.csv"]:
        if os.path.exists(csv_path):
            try:
                csv_df = pd.read_csv(csv_path)
                csv_df.columns = [c.lower() for c in csv_df.columns]
                if {"timestamp", "high", "low"}.issubset(csv_df.columns):
                    df = pd.concat([df, csv_df[["timestamp", "high", "low"]]], ignore_index=True)
            except Exception:
                pass

    if df is None or df.empty:
        raise RuntimeError("No bars found in DB or CSV.")

    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "high", "low"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def analyze_best_1h_windows():
    df = load_bars()

    # 1-Hour (12 x 5-min bars) forward rolling window
    idx_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    df["high_1h"] = df["high"].rolling(window=idx_1h).max()
    df["low_1h"] = df["low"].rolling(window=idx_1h).min()
    df["range_1h"] = df["high_1h"] - df["low_1h"]

    # Filter out incomplete trailing windows or session/weekend gaps (> 70 mins across 12 bars)
    df["span_mins"] = (df["timestamp"].shift(-11) - df["timestamp"]).dt.total_seconds() / 60.0
    valid = df.iloc[:-12].copy()
    valid = valid[valid["span_mins"] <= 70].dropna(subset=["range_1h"])

    valid["cairo_hour"] = valid["timestamp"].dt.tz_convert("Africa/Cairo").dt.hour

    summary = (
        valid.groupby("cairo_hour")["range_1h"]
        .agg(
            avg_pts="mean",
            med_50_pts="median",
            p80_pts=lambda x: x.quantile(0.80),
            p90_pts=lambda x: x.quantile(0.90),
        )
        .reset_index()
        .sort_values(by="avg_pts", ascending=True)
    )

    print("=" * 78)
    print("BEST 1-HOUR WINDOWS & POINT THRESHOLDS (CAIRO TIME)")
    print("=" * 78)
    print(f"{'Cairo Time':<12} | {'Avg 1H Pts':<11} | {'50% <= Pts':<11} | {'80% <= Pts':<11} | {'90% <= Pts':<11}")
    print("-" * 78)
    for _, row in summary.iterrows():
        print(
            f"{int(row['cairo_hour']):02d}:00 Cairo | "
            f"{row['avg_pts']:>9.0f}   | "
            f"{row['med_50_pts']:>9.0f}   | "
            f"{row['p80_pts']:>9.0f}   | "
            f"{row['p90_pts']:>9.0f}"
        )
    print("=" * 78)

if __name__ == "__main__":
    analyze_best_1h_windows()
