import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram
from scipy.stats import pearsonr

# 1. Membaca data dari CSV
data = pd.read_csv('dataset/custom/tarakan_weather_station_2023_smoothed1.csv')  # Ganti dengan nama file CSV Anda
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

# 3. Filter data untuk rentang 1 bulan (Januari 2023)
start_date = '2023-01-01 00:00:00'
end_date = '2023-01-31 23:00:00'
data = data.loc[start_date:end_date]

# 4. Pemetaan nama kolom ke header yang diinginkan
column_mapping = {
    'dewpt': 'Titik embun',
    'temp': 'Suhu',
    'wetb': 'Bola basah',
    'wddir': 'Arah angin',
    'wdsp': 'Kecepatan angin',
    'rhum': 'Kelembaban',
    'msl': 'Tekanan udara'
}

# 5. Fungsi untuk uji periodisitas dan perbandingan dengan sinyal sinusoidal
def compare_with_sinusoidal(series, col_name, subplot_pos):
    f, Pxx = periodogram(series)
    peak_freq = f[np.argmax(Pxx)]
    period_est = 1 / peak_freq if peak_freq > 0 else len(series) / 2  # Estimasi periode dalam poin
    
    t = np.arange(len(series))
    amplitude = series.std()  # Amplitudo berdasarkan standar deviasi data
    sinusoidal = amplitude * np.sin(2 * np.pi * t / period_est) + series.mean()  # Sinyal sinusoidal sederhana

    # Plot dalam subplot
    plt.subplot(7, 1, subplot_pos)
    plt.plot(series.index, series, label='Original Data', color='#A9A9A9', linewidth=0.8)  # Abu-abu lembut
    plt.plot(series.index, sinusoidal[:len(series)], label='Sinusoidal Fit', color='#FF4040', linestyle='--', linewidth=0.8)
    plt.title(column_mapping[col_name])  # Gunakan nama yang dimapping
    plt.legend()
    if subplot_pos == 7:  # Hanya tambahkan label sumbu x pada subplot terakhir
        plt.xlabel('Time')
    plt.ylabel('Value')

# 6. Membuat figure tunggal dengan semua subplot
plt.figure(figsize=(15, 15))  # Ukuran figure disesuaikan untuk 7 subplot

# 7. Menguji setiap kolom dalam satu figure
for i, col in enumerate(columns, 1):
    series = data[col].dropna()
    compare_with_sinusoidal(series, col, i)

# 8. Atur tata letak dan simpan
plt.tight_layout()
plt.savefig('combined_sinusoidal_comparison_jan_2023.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 9. Hitung dan cetak korelasi untuk setiap kolom
for col in columns:
    series = data[col].dropna()
    f, Pxx = periodogram(series)
    peak_freq = f[np.argmax(Pxx)]
    period_est = 1 / peak_freq if peak_freq > 0 else len(series) / 2
    t = np.arange(len(series))
    amplitude = series.std()
    sinusoidal = amplitude * np.sin(2 * np.pi * t / period_est) + series.mean()
    correlation, p_value = pearsonr(series, sinusoidal[:len(series)])
    print(f"\nAnalisis untuk {column_mapping[col]}:")
    print(f"Estimasi Periode: {period_est:.0f} poin (sekitar {period_est:.0f} jam jika data per jam)")  # Asumsi per jam
    print(f"Korelasi dengan Sinyal Sinusoidal: {correlation:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05 and abs(correlation) > 0.5:
        print("Kesimpulan: Terdapat periodisitas signifikan yang mirip dengan sinyal sinusoidal.")
    else:
        print("Kesimpulan: Tidak ada periodisitas signifikan yang mirip dengan sinyal sinusoidal.")