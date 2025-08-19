import pandas as pd
import os

def preprocess_tarakan(path, save_path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset='date')
    df = df.sort_values('date')

    # Interpolasi nilai yang hilang (bukan fillna)
    df.interpolate(method='linear', inplace=True)

    # Untuk jaga-jaga jika masih ada NaN di awal/akhir
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Simpan hasilnya
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved cleaned data to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to raw CSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to save cleaned CSV")
    args = parser.parse_args()

    preprocess_tarakan(args.input, args.output)
