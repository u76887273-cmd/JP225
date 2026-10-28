import os
import sys
import psycopg
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_URL = os.environ.get("DATABASE_URL")

def load_bars():
    df = pd.DataFrame()

    # 1. Try using train.py's built-in loader first (matches your repo's exact table & CSV logic)
    try:
        from train import sync_and_load_all_m5_bars
        df = sync_and_load_all_m5_bars()
        if df is not None and not df.empty:
            print(f"Loaded {len(df):,} JP225 bars via train.sync_and_load_all_m5_bars()")
    except Exception as e:
        print(f"Note on train loader: {e}")

    # 2. If still empty and DATABASE_URL is available, auto-discover the OHLC table in Supabase
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
                    print(f"Discovered OHLC tables in DB: {tables}")

                    best_rows = []
                    best_table = None
                    for t in tables:
                        cur.execute(f"SELECT timestamp, high, low FROM {t} ORDER BY timestamp ASC;")
                        rows = cur.fetchall()
                        if len(rows) > len(best_rows):
                            best_rows = rows
                            best_table = t

                    if best_rows:
                        print(f"Loaded {len(best_rows):,} bars from '{best_table}'")
                        df = pd.DataFrame(best_rows, columns=["timestamp", "high", "low"])
        except Exception as e:
            print(f"Note on DB auto-discovery: {e}")

    # 3. Always also check local CSV files and combine so no bars are missed
    for csv_path in ["data_m5.csv", "src/data_m5.csv", "jp225_m5.csv"]:
        if os.path.exists(csv_path):
            try:
                csv_df = pd.read_csv(csv_path)
                csv_df.columns = [c.lower() for c in csv_df.columns]
                if {"timestamp", "high", "low"}.issubset(csv_df.columns):
                    print(f"Loaded {len(csv_df):,} bars from local '{csv_path}'")
                    df = pd.concat([df, csv_df[["timestamp", "high", "low"]]], ignore_index=True)
            except Exception as e:
                print(f"Note reading {csv_path}: {e}")

    if df is None or df.empty:
        raise RuntimeError("No bars found in DB or CSV. Make sure DATABASE_URL is passed in workflow env.")

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

def analyze_quiet_hours():
    df = load_bars()

    # 1-Hour (12 bars) and 2-Hour (24 bars) forward rolling windows
    idx_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    idx_2h = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)

    df["high_1h"] = df["high"].rolling(window=idx_1h).max()
    df["low_1h"] = df["low"].rolling(window=idx_1h).min()
    df["range_1h"] = df["high_1h"] - df["low_1h"]

    df["high_2h"] = df["high"].rolling(window=idx_2h).max()
    df["low_2h"] = df["low"].rolling(window=idx_2h).min()
    df["range_2h"] = df["high_2h"] - df["low_2h"]

    # Drop trailing incomplete windows and weekend/session gaps (> 2h15m across 24 bars)
    df["span_mins"] = (df["timestamp"].shift(-23) - df["timestamp"]).dt.total_seconds() / 60.0
    valid = df.iloc[:-24].copy()
    valid = valid[valid["span_mins"] <= 135].dropna(subset=["range_1h", "range_2h"])

    valid["cairo_hour"] = valid["timestamp"].dt.tz_convert("Africa/Cairo").dt.hour

    # Calculate percentages for staying under 100 points
    valid["1h_under_100"] = (valid["range_1h"] <= 100.0).astype(float) * 100.0
    valid["2h_under_100"] = (valid["range_2h"] <= 100.0).astype(float) * 100.0

    summary = (
        valid.groupby("cairo_hour")
        .agg(
            avg_1h=("range_1h", "mean"),
            avg_2h=("range_2h", "mean"),
            pct_1h_under_100=("1h_under_100", "mean"),
            pct_2h_under_100=("2h_under_100", "mean"),
        )
        .reset_index()
        .sort_values(by=["pct_2h_under_100", "avg_2h"], ascending=[False, True])
    )

    print("=" * 76)
    print("JP225 QUIETEST HOURS (CAIRO TIME)")
    print("=" * 76)
    print(f"{'Cairo Time':<12} | {'Avg 1H Pts':<12} | {'Avg 2H Pts':<12} | {'1H <= 100':<11} | {'2H <= 100':<11}")
    print("-" * 76)
    for _, row in summary.iterrows():
        print(
            f"{int(row['cairo_hour']):02d}:00 Cairo | "
            f"{row['avg_1h']:>10.1f} | "
            f"{row['avg_2h']:>10.1f} | "
            f"{row['pct_1h_under_100']:>9.1f}% | "
            f"{row['pct_2h_under_100']:>9.1f}%"
        )
    print("=" * 76)

if __name__ == "__main__":
    analyze_quiet_hours()
