import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import time
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Trading Pro", layout="wide")

# ==========================================
# SIDEBAR AFILIASI (MONETISASI)
# ==========================================
st.sidebar.title("🤝 Mitra Resmi")
st.sidebar.markdown("""
**Siap Mencetak Profit Hari Ini?** 🚀

Jangan biarkan hasil analisis AI ini sia-sia! Langsung eksekusi pembelian saham Anda di aplikasi sekuritas berizin OJK. 

Dapatkan **BONUS SALDO atau KOIN GRATIS** khusus pengguna baru dengan mendaftar melalui tautan di bawah ini:

📈 **[Daftar Ajaib Sekuritas](https://ajaib.co.id/)**
*(Gunakan kode referral: **KODEMU-123**)*

💼 **[Daftar Stockbit](https://stockbit.com/)**
*(Gunakan kode referral: **KODEMU-456**)*

---
*Catatan: Kami akan menerima komisi apresiasi jika Anda mendaftar menggunakan tautan di atas, tanpa ada potongan biaya apapun dari saldo Anda.*
""")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 AI Trading Pro")

# ==========================================
# HALAMAN UTAMA
# ==========================================
st.title("📈 AI Trading Pro: LQ45 Screener & Full BEI Analysis")
st.markdown("Aplikasi pintar ini dilengkapi dengan **Auto-Screener (Indeks LQ45)**, **Data Fundamental**, Grafik Candlestick, Prediksi AI, dan Backtesting Profit.")

st.warning("""
**⚠️ DISCLAIMER HUKUM & RISIKO FINANSIAL:**
Semua data, analisis, dan prediksi AI yang ditampilkan di aplikasi ini hanya bertujuan sebagai **alat bantu edukasi dan informasi**. 
Ini **BUKAN** merupakan saran keuangan, rekomendasi investasi, atau ajakan pasti untuk membeli/menjual saham tertentu. 
Pasar saham memiliki risiko tinggi, dan Anda bertanggung jawab penuh atas segala keputusan finansial serta kerugian yang mungkin terjadi.
""")

# Daftar 45 Saham Paling Likuid di BEI (Indeks LQ45)
LQ45_TICKERS = [
    "ACES.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBNI.JK", "BBCA.JK",
    "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "BRPT.JK", "BUKA.JK", "CPIN.JK", "EMTK.JK", "ESSA.JK", "EXCL.JK",
    "GGRM.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK", "INTP.JK", "ITMG.JK", "KLBF.JK",
    "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "MTEL.JK", "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
    "SRTG.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK"
]

# --- FUNGSI UTAMA AI ---
def hitung_indikator_dan_prediksi(data, model_choice):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily_Return'].rolling(window=10).std()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * std_dev
    data['BB_Lower'] = data['BB_Middle'] - 2 * std_dev
    
    data['Target'] = np.where(data['Close'].shift(-5) > data['Close'], 1, 0)
    data_clean = data.dropna()
    
    if len(data_clean) < 50:
        return None
        
    features = ['SMA_10', 'SMA_50', 'Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    X = data_clean[features]
    y = data_clean['Target']
    
    split_idx = int(len(data_clean) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        latest_data = X.iloc[-1:]
        future_pred = model.predict(latest_data)[0]
    else:
        model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        latest_data_scaled = scaler.transform(X.iloc[-1:])
        future_pred = model.predict(latest_data_scaled)[0]
        
    accuracy = accuracy_score(y_test, predictions)
    return data_clean, future_pred, accuracy, predictions, split_idx, features

def dapatkan_rekomendasi(future_pred, last_rsi):
    if future_pred == 1 and last_rsi < 70:
        return "🟢 BELI", "Potensi Naik & Harga Wajar", 1
    elif future_pred == 1 and last_rsi >= 70:
        return "🟡 TAHAN", "Potensi Naik, Tapi Status Overbought", 0
    elif future_pred == 0 and last_rsi > 30:
        return "🔴 JUAL", "Potensi Turun & Momentum Melemah", -1
    else:
        return "⚪ PANTAU", "Status Oversold / Terlalu Murah", 0


# --- TABS ---
tab_screener, tab_individu = st.tabs(["🚀 Auto-Screener LQ45 (Top 45 BEI)", "🔍 Analisis Bebas Seluruh Saham (900+)"])

# ==========================================
# TAB 1: SCREENER (LQ45)
# ==========================================
with tab_screener:
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("AI akan mengecek 45 saham paling aktif di bursa secara bersamaan untuk mencari sinyal beli terbaik.")
    
    col_scr1, col_scr2 = st.columns([1, 3])
    with col_scr1:
        screener_model = st.selectbox("Model AI untuk Screener", ["Random Forest", "Deep Learning (Neural Network)"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Mulai Pindai 45 Saham (Scan Market)", type="primary", use_container_width=True):
        hasil_scan = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_saham = len(LQ45_TICKERS)
        
        for i, kode in enumerate(LQ45_TICKERS):
            status_text.text(f"Memindai {kode} ({i+1}/{total_saham})... Mohon tunggu sekitar 1-2 menit.")
            try:
                df_scan = yf.download(kode, period="2y", progress=False)
                if df_scan.empty: continue
                if isinstance(df_scan.columns, pd.MultiIndex):
                    df_scan.columns = [c[0] for c in df_scan.columns]
                
                hasil = hitung_indikator_dan_prediksi(df_scan, screener_model)
                if hasil is None: continue
                
                data_clean, future_pred, accuracy, predictions, split_idx, features = hasil
                last_rsi = data_clean['RSI_14'].iloc[-1]
                last_close = data_clean['Close'].iloc[-1]
                
                aksi, alasan, skor = dapatkan_rekomendasi(future_pred, last_rsi)
                
                hasil_scan.append({
                    "Kode Saham": kode.replace('.JK', ''),
                    "Harga": f"Rp {last_close:,.0f}",
                    "Akurasi AI": f"{accuracy*100:.1f}%",
                    "Prediksi Tren": "NAIK 📈" if future_pred == 1 else "TURUN 📉",
                    "RSI": round(last_rsi, 1),
                    "Rekomendasi": aksi,
                    "Keterangan": alasan,
                    "_Skor": skor 
                })
            except Exception as e:
                pass
            
            time.sleep(0.1)
            progress_bar.progress((i + 1) / total_saham)
        
        status_text.text("✅ Pemindaian Selesai! Berikut adalah peringkat saham LQ45 hari ini:")
        df_hasil = pd.DataFrame(hasil_scan)
        if not df_hasil.empty:
            df_hasil = df_hasil.sort_values(by=["_Skor", "Akurasi AI"], ascending=[False, False]).drop(columns=["_Skor"]).reset_index(drop=True)
            df_hasil.index = df_hasil.index + 1  # Ubah indeks mulai dari 1
            df_hasil.index.name = "Rank"         # Beri nama kolom indeks
            st.dataframe(df_hasil, use_container_width=True)
        else:
            st.warning("Gagal menarik data pasar (Pastikan koneksi internet stabil).")


# ==========================================
# TAB 2: ANALISIS BEBAS (SEMUA SAHAM)
# ==========================================
with tab_individu:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Teks diperpendek agar sejajar dan rapi
        ticker_input = st.text_input("Kode Saham (Contoh: BBCA, BREN)", value="BBCA")
        ticker = ticker_input.strip().upper()
        # Otomatis menambahkan .JK di belakang layar
        if not ticker.endswith(".JK") and len(ticker) == 4:
            ticker = ticker + ".JK"
            
    with col2:
        period = st.selectbox("Periode Waktu", ["1y", "2y", "5y", "10y"], index=2)
    with col3:
        model_choice = st.selectbox("Algoritma AI", ["Random Forest", "Deep Learning (Neural Network)"], key="ind_model")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol dilebarkan agar simetris
    if st.button("Jalankan Analisis Individual", type="primary", use_container_width=True):
        with st.spinner(f"Menarik Data Fundamental & Harga {ticker}..."):
            saham_obj = yf.Ticker(ticker)
            info = saham_obj.info
            data = saham_obj.history(period=period)
            
            if data.empty:
                st.error(f"Data harga untuk {ticker} tidak ditemukan. Pastikan kodenya benar.")
            else:
                st.success("Data berhasil diunduh!")
                
                st.markdown("### 🏢 Data Fundamental Perusahaan")
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                
                mcap = info.get("marketCap", "N/A")
                if mcap != "N/A": mcap = f"Rp {mcap/1e12:,.1f} Triliun"
                
                pe_ratio = info.get("trailingPE", "N/A")
                if pe_ratio != "N/A": pe_ratio = f"{pe_ratio:.2f}x"
                
                div_yield = info.get("dividendYield", "N/A")
                if div_yield != "N/A" and div_yield is not None: 
                    div_yield = f"{div_yield*100:.2f}%"
                else:
                    div_yield = "N/A"
                    
                sektor = info.get("sector", "N/A")
                
                col_f1.metric("Market Cap (Kapitalisasi)", mcap)
                col_f2.metric("P/E Ratio (Valuasi)", pe_ratio)
                col_f3.metric("Dividend Yield", div_yield)
                col_f4.metric("Sektor Industri", sektor)
                
                st.markdown("---")
                
                st.subheader(f"📊 Grafik Candlestick {ticker}")
                
                data['SMA_10'] = data['Close'].rolling(window=10).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'],
                                name="Candlestick"))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_10'], line=dict(color='orange', width=1.5), name='SMA 10'))
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='blue', width=1.5), name='SMA 50'))
                fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0),
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                with st.spinner('Memproses Kecerdasan Buatan...'):
                    hasil = hitung_indikator_dan_prediksi(data, model_choice)
                    if hasil is not None:
                        data_clean, future_pred, accuracy, predictions, split_idx, features = hasil
                        
                        test_dates = data_clean.index[split_idx:]
                        actual_returns = data_clean['Daily_Return'].iloc[split_idx:]
                        signals = pd.Series(predictions, index=test_dates).shift(1).fillna(0)
                        
                        strategy_returns = actual_returns * signals
                        cumulative_stock = (1 + actual_returns).cumprod() - 1
                        cumulative_strategy = (1 + strategy_returns).cumprod() - 1
                        
                        st.subheader("💰 Simulasi Keuntungan AI (Backtesting)")
                        col_b1, col_b2 = st.columns(2)
                        col_b1.metric("Beli & Diam Biasa", f"{cumulative_stock.iloc[-1] * 100:.2f}%")
                        col_b2.metric("Trading Mengikuti AI", f"{cumulative_strategy.iloc[-1] * 100:.2f}%", 
                                      delta=f"{(cumulative_strategy.iloc[-1] - cumulative_stock.iloc[-1]) * 100:.2f}% vs Beli Biasa")
                        
                        profit_df = pd.DataFrame({'Beli & Diam Biasa': cumulative_stock * 100, 'Mengikuti AI': cumulative_strategy * 100})
                        st.line_chart(profit_df)
                        
                        last_rsi = data_clean['RSI_14'].iloc[-1]
                        aksi, alasan, _ = dapatkan_rekomendasi(future_pred, last_rsi)
                        
                        st.markdown("---")
                        st.subheader("💡 Kesimpulan Sinyal Hari Ini")
                        st.info(f"**Rekomendasi AI:** {aksi} ({alasan})")
                        st.write(f"*Akurasi histori model {model_choice} ini adalah {accuracy*100:.1f}%.*")
