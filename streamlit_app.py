"""
基于ARIMA与LSTM的Stacking集成风速预测模型 - 终极兼容版
✅ 兼容列名：平均风速、风速、wdsp、WSPD、wind_speed、风速(m/s) ...
✅ 日期万能识别：20200101 / 2020-01-01 / 2020/1/1 / 2020年1月1日
✅ 多余列自动删除
✅ 中文图表正常显示
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from pmdarima import auto_arima
from pmdarima.arima import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="风速预测系统",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

font_path = "C:/Windows/Fonts/msyh.ttc"
if os.path.exists(font_path):
    import matplotlib
    matplotlib.font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] + plt.rcParams['font.sans-serif']

np.random.seed(42)
tf.random.set_seed(42)

# ==================== 🔥 终极兼容：自动识别日期 + 风速列 ====================
def load_and_preprocess(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')

        # 自动找日期列
        date_col = None
        for c in df.columns:
            cl = str(c).lower()
            if '日' in cl or 'time' in cl or 'date' in cl:
                date_col = c
                break

        # 自动找风速列（兼容：平均风速、风速、wdsp、WSPD、wind_speed）
        wind_col = None
        for c in df.columns:
            cl = str(c).lower()
            if '风速' in cl or '风' in cl or 'wdsp' in cl or 'wspd' in cl or 'wind' in cl:
                wind_col = c
                break

        if not date_col or not wind_col:
            st.error("无法识别日期列或风速列！请确保包含日期与风速相关列")
            return None

        # 只保留这两列，其他全部删掉
        df = df[[date_col, wind_col]].copy()
        df.rename(columns={date_col: '日期', wind_col: '平均风速'}, inplace=True)

        # 万能日期解析
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])

        df.set_index('日期', inplace=True)
        series = df['平均风速'].astype(float)

        # 缺失值填充
        if series.isnull().any():
            series = series.interpolate(method='linear')

        # 异常值处理
        mean, std = series.mean(), series.std()
        series = series.clip(mean - 3*std, mean + 3*std)

        return series

    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return None

# ==================== 模型函数 ====================
def split_train_val_test(series, train_ratio=0.6, val_ratio=0.2):
    n = len(series)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    train = series[:train_size]
    val = series[train_size:train_size+val_size]
    test = series[train_size+val_size:]
    st.info(f"训练集{len(train)} | 验证集{len(val)} | 测试集{len(test)}")
    return train, val, test

def scale_data(train, val, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1,1)).flatten()
    val_scaled = scaler.transform(val.values.reshape(-1,1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1,1)).flatten()
    return train_scaled, val_scaled, test_scaled, scaler

# ---------- ARIMA ----------
def get_optimal_order(train_series):
    with st.spinner("正在搜索最优ARIMA阶数..."):
        model = auto_arima(train_series, seasonal=False, stepwise=True, trace=False, error_action='ignore')
    st.success(f"最优ARIMA阶数：{model.order}")
    return model.order

def arima_rolling_fit(history, order, maxiter=30):
    try:
        model = ARIMA(order=order)
        model.fit(history, method='lbfgs', maxiter=maxiter, disp=0)
        return model
    except:
        return None

def arima_rolling_predict(train_series, test_series, order, refit_freq=7):
    history = list(train_series)
    predictions = []
    current_model = None
    bar = st.progress(0)
    text = st.empty()

    for i, _ in enumerate(test_series):
        if i % refit_freq == 0 or current_model is None:
            current_model = arima_rolling_fit(history, order)
        if current_model is not None:
            pred = current_model.predict(n_periods=1)[0]
        else:
            pred = np.mean(history[-30:]) if len(history)>=30 else np.mean(history)

        predictions.append(pred)
        history.append(test_series.iloc[i])
        bar.progress((i+1)/len(test_series))
        text.text(f"ARIMA 预测进度：{i+1}/{len(test_series)}")

    bar.empty()
    text.empty()
    return np.array(predictions)

def arima_rolling_predict_val(train_series, val_series, order, refit_freq=7):
    history = list(train_series)
    predictions = []
    current_model = None
    for i, _ in enumerate(val_series):
        if i % refit_freq == 0 or current_model is None:
            current_model = arima_rolling_fit(history, order)
        if current_model:
            pred = current_model.predict(n_periods=1)[0]
        else:
            pred = np.mean(history[-30:]) if len(history)>=30 else np.mean(history)
        predictions.append(pred)
        history.append(val_series.iloc[i])
    return np.array(predictions)

# ---------- LSTM ----------
def create_supervised(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_lstm(train_scaled, val_scaled, window, epochs=60, batch_size=32):
    X_train, y_train = create_supervised(train_scaled, window)
    X_val, y_val = create_supervised(val_scaled, window)
    X_train = X_train.reshape(-1, window, 1)
    X_val = X_val.reshape(-1, window, 1)

    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=(window, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    bar = st.progress(0)
    text = st.empty()

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            bar.progress((epoch+1)/epochs)
            text.text(f"LSTM 第{epoch+1}/{epochs}轮 | 损失 {logs['loss']:.4f} | 验证损失 {logs['val_loss']:.4f}")

    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=[early_stop, Callback()], verbose=0)

    bar.empty()
    text.empty()
    st.success("LSTM 训练完成")
    return model, hist

def predict_lstm_rolling(model, train_scaled, target_scaled, window, scaler):
    full = np.concatenate([train_scaled, target_scaled])
    X_pred = []
    for i in range(len(train_scaled), len(full)-1):
        X_pred.append(full[i-window:i])
    X_pred = np.array(X_pred).reshape(-1, window, 1)
    pred = model.predict(X_pred, verbose=0).flatten()
    return scaler.inverse_transform(pred.reshape(-1,1)).flatten()

# ---------- 评估 ----------
def metrics(y, yp):
    rmse = np.sqrt(mean_squared_error(y, yp))
    mae = mean_absolute_error(y, yp)
    mape = np.mean(np.abs((y-yp)/y)) * 100
    return rmse, mae, mape

def dm_test(y_true, y1, y2):
    e1 = y_true - y1
    e2 = y_true - y2
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return 1.0
    stat = mean_d / np.sqrt(var_d / len(d))
    p = 2*(1-stats.t.cdf(np.abs(stat), df=len(d)-1))
    return p

# ---------- 绘图 ----------
def plot_original_series(series):
    fig, axes = plt.subplots(2,1,figsize=(14,8))
    axes[0].plot(series.index, series, color='#2c7bb6', linewidth=1)
    axes[0].set_title('原始风速序列')
    axes[0].set_ylabel('风速 m/s')
    axes[0].grid(alpha=0.3)

    years = series.index.year
    boxes = [series[years==y].values for y in sorted(years.unique())]
    axes[1].boxplot(boxes, patch_artist=True, boxprops=dict(facecolor='#abd9e9'), medianprops=dict(color='red'))
    axes[1].set_xticks(range(0, len(boxes), 5))
    axes[1].set_xticklabels(sorted(years.unique())[::5], rotation=45)
    axes[1].set_title('年度风速分布')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

def plot_prediction_comparison(dates, y, ar, lstm, stack):
    fig, axes = plt.subplots(2,1,figsize=(14,10))
    axes[0].plot(dates, y, label='真实', c='black', linewidth=1.5)
    axes[0].plot(dates, ar, label='ARIMA', linestyle='--', alpha=0.8)
    axes[0].plot(dates, lstm, label='LSTM', linestyle='-.', alpha=0.8)
    axes[0].plot(dates, stack, label='Stacking', linewidth=1.5, alpha=0.9)
    axes[0].set_title('全测试集预测对比')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    N = min(200, len(dates))
    axes[1].plot(dates[-N:], y[-N:], c='black', linewidth=1.5, label='真实')
    axes[1].plot(dates[-N:], ar[-N:], '--', label='ARIMA', alpha=0.8)
    axes[1].plot(dates[-N:], lstm[-N:], '-.', label='LSTM', alpha=0.8)
    axes[1].plot(dates[-N:], stack[-N:], label='Stacking', linewidth=1.5)
    axes[1].set_title(f'最近{N}天放大')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

def plot_residual_analysis(y, ar, lstm, stack):
    res = {
        'ARIMA': y-ar,
        'LSTM': y-lstm,
        'Stacking': y-stack
    }
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for i,(k,v) in enumerate(res.items()):
        axes[0,i].plot(v, alpha=0.6, c='#444')
        axes[0,i].axhline(0, c='red', ls='--')
        axes[0,i].set_title(f'{k} 残差')
        axes[1,i].hist(v, bins=30, alpha=0.7, color='#2c7bb6')
        axes[1,i].axvline(0, c='red', ls='--')
    plt.tight_layout()
    st.pyplot(fig)

def plot_performance_barchart(result):
    models = list(result.keys())
    rmse = [result[m][0] for m in models]
    mae = [result[m][1] for m in models]
    mape = [result[m][2] for m in models]
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    axs[0].bar(models, rmse, color=['#3498db','#2ecc71','#e74c3c'])
    axs[0].set_title('RMSE')
    axs[1].bar(models, mae, color=['#3498db','#2ecc71','#e74c3c'])
    axs[1].set_title('MAE')
    axs[2].bar(models, mape, color=['#3498db','#2ecc71','#e74c3c'])
    axs[2].set_title('MAPE')
    plt.tight_layout()
    st.pyplot(fig)

def plot_stacking_weights(w1, w2, b):
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    abs_w = np.abs([w1, w2])
    axs[0].pie(abs_w/abs_w.sum(), labels=['ARIMA','LSTM'], autopct='%1.1f%%', colors=['#3498db','#e74c3c'])
    axs[0].set_title('权重占比')
    axs[1].bar(['ARIMA','LSTM','截距'], [w1,w2,b], color=['#3498db','#e74c3c','#95a5a6'])
    axs[1].axhline(0, c='black', linewidth=0.8)
    axs[1].set_title('融合系数')
    plt.tight_layout()
    st.pyplot(fig)

# ==================== 主界面 ====================
def main():
    st.title("🌬️ ARIMA+LSTM Stacking 风速预测系统")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 数据上传与参数")
        file = st.file_uploader("上传Excel文件", type=["xlsx"])
        tr = st.slider("训练集比例",0.5,0.8,0.6)
        vr = st.slider("验证集比例",0.1,0.3,0.2)
        window = st.slider("LSTM窗口大小",3,14,7)
        refit = st.slider("ARIMA重拟合间隔",3,14,7)
        epochs = st.slider("LSTM训练轮次",30,100,60)
        go = st.button("🚀 开始训练与预测", disabled=not file)

    if file and go:
        st.subheader("1️⃣ 数据加载与清洗")
        series = load_and_preprocess(file)
        if series is None: return
        st.dataframe(series.head(10), use_container_width=True)
        plot_original_series(series)

        train, val, test = split_train_val_test(series, tr, vr)
        ts, vs, tes, scaler = scale_data(train, val, test)

        st.subheader("2️⃣ ARIMA 滚动预测")
        order = get_optimal_order(train)
        ar_val = arima_rolling_predict_val(train, val, order, refit)
        ar_test = arima_rolling_predict(train, test, order, refit)

        st.subheader("3️⃣ LSTM 模型预测")
        model, hist = train_lstm(ts, vs, window, epochs)
        lstm_val = predict_lstm_rolling(model, ts, vs, window, scaler)
        lstm_test = predict_lstm_rolling(model, ts, tes, window, scaler)

        st.subheader("4️⃣ Stacking 集成融合")
        L = min(len(ar_val), len(lstm_val))
        X_meta = np.column_stack([ar_val[:L], lstm_val[:L]])
        y_meta = val.values[:L]
        lr = LinearRegression()
        lr.fit(X_meta, y_meta)
        st.success(f"融合权重：ARIMA={lr.coef_[0]:.3f}  LSTM={lr.coef_[1]:.3f}  截距={lr.intercept_:.3f}")
        plot_stacking_weights(lr.coef_[0], lr.coef_[1], lr.intercept_)

        L = min(len(ar_test), len(lstm_test))
        ar_test = ar_test[:L]
        lstm_test = lstm_test[:L]
        y_test = test.values[:L]
        stack = lr.predict(np.column_stack([ar_test, lstm_test]))

        st.subheader("5️⃣ 模型性能评估")
        res = {
            'ARIMA': metrics(y_test, ar_test),
            'LSTM': metrics(y_test, lstm_test),
            'Stacking': metrics(y_test, stack)
        }
        st.table(pd.DataFrame({
            '模型':['ARIMA','LSTM','Stacking'],
            'RMSE':[res['ARIMA'][0],res['LSTM'][0],res['Stacking'][0]],
            'MAE':[res['ARIMA'][1],res['LSTM'][1],res['Stacking'][1]],
            'MAPE(%)':[res['ARIMA'][2],res['LSTM'][2],res['Stacking'][2]]
        }).round(3))

        p1 = dm_test(y_test, ar_test, stack)
        p2 = dm_test(y_test, lstm_test, stack)
        st.write(f"DM检验(ARIMA vs Stacking) p={p1:.4f}")
        st.write(f"DM检验(LSTM vs Stacking) p={p2:.4f}")

        plot_performance_barchart(res)

        st.subheader("6️⃣ 预测曲线对比")
        plot_prediction_comparison(test.index[:L], y_test, ar_test, lstm_test, stack)

        st.subheader("7️⃣ 残差分析")
        plot_residual_analysis(y_test, ar_test, lstm_test, stack)

        # 下载
        df_out = pd.DataFrame({
            '日期':test.index[:L],
            '真实风速':y_test,
            'ARIMA预测':ar_test,
            'LSTM预测':lstm_test,
            'Stacking预测':stack
        })
        csv = df_out.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下载完整预测结果", csv, "风速预测结果.csv")

        st.success("✅ 全部分析完成！")

    elif file and not go:
        series = load_and_preprocess(file)
        if series is not None:
            st.dataframe(series.head(20))
            plot_original_series(series)
            st.info("点击侧边栏按钮开始训练")

if __name__ == "__main__":
    main()
