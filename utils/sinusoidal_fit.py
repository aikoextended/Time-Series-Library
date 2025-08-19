import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy.signal import periodogram
from scipy.stats import pearsonr

# 1. Membaca data dari CSV
data = pd.read_csv('dataset/custom/tarakan_weather_station_2023_cleaned.csv')  # Ganti dengan nama file CSV Anda
columns = [
    'dewpt',
    'temp',
    'wetb',
    'wddir',
    'wdsp',
    'rhum',
    'msl'
]

# 2. Mengatur indeks waktu
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S')
data.set_index('date', inplace=True)

# 3. Fungsi untuk uji periodisitas dan perbandingan dengan sinyal sinusoidal
def compare_with_sinusoidal(series, col_name):
    plt.figure(figsize=(15, 10))

    f, Pxx = periodogram(series)
    peak_freq = f[np.argmax(Pxx)]
    period_est = 1 / peak_freq if peak_freq > 0 else len(series) / 2  # Estimasi periode
    
    t = np.arange(len(series))
    amplitude = series.std()  # Amplitudo berdasarkan standar deviasi data
    sinusoidal = amplitude * np.sin(2 * np.pi * t / period_est) + series.mean()  # Sinyal sinusoidal sederhana

    # e. Plot perbandingan
    plt.figure(figsize=(15, 4))
    plt.plot(series.index, series, label='Original Data', color='b', linewidth=0.8)
    plt.plot(series.index, sinusoidal[:len(series)], label='Sinusoidal Fit', color='r', linestyle='--', linewidth=0.8)
    plt.title(f'Comparison with Sinusoidal - {col_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    #plt.savefig(f'{col_name}_sinusoidal_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # f. Hitung korelasi untuk mengukur kesesuaian
    correlation, p_value = pearsonr(series, sinusoidal[:len(series)])
    print(f"\nAnalisis untuk {col_name}:")
    print(f"Estimasi Periode: {period_est:.0f} poin (sekitar {period_est * 15 / 60:.1f} jam jika 15 menit per interval)")
    print(f"Korelasi dengan Sinyal Sinusoidal: {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05 and abs(correlation) > 0.5:
        print("Kesimpulan: Terdapat periodisitas signifikan yang mirip dengan sinyal sinusoidal.")
    else:
        print("Kesimpulan: Tidak ada periodisitas signifikan yang mirip dengan sinyal sinusoidal.")

# 4. Menguji setiap kolom
for col in columns:
    series = data[col].dropna()
    compare_with_sinusoidal(series, col)