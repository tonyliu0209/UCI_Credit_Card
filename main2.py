import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# 設定頁面配置
st.set_page_config(layout="wide", page_title="信用卡預測模型", initial_sidebar_state="expanded")

# 標題
st.title("信用卡違約預測系統")

# 建立左右欄位
col_left, col_right = st.columns([1, 2])

# ========== 左側：模型選擇 ==========
with col_left:
    with st.expander("🎯 模型選擇", expanded=True):
        model_option = st.selectbox(
            "選擇預測模型",
            ["KNN", "LogisticRegression", "隨機森林", "XGBoost"]
        )

# ========== 右側：資料展示和預測 ==========
with col_right:
    # 讀取資料
    df = pd.read_csv("UCI_Credit_Card.csv")
    
    # 顯示前10筆資料
    st.subheader("資料概覽（前10筆）")
    st.dataframe(df.head(10))
    
    # 準備 X 和 y
    X = df.drop(columns=["ID", "default.payment.next.month"])
    y = df["default.payment.next.month"]
    
    # 顯示 y 的各分類資料數
    st.subheader("違約情況統計")
    y_counts = y.value_counts().sort_index()
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("未違約（0）", y_counts.get(0, 0))
    with col_stat2:
        st.metric("違約（1）", y_counts.get(1, 0))
    
    # 顯示統計圖
    st.bar_chart(y_counts)

# ========== 預測部分 ==========
st.divider()
st.subheader("隨機預測")

col_predict1, col_predict2 = st.columns([1, 2])

with col_predict1:
    if st.button("🎲 隨機抽選並預測", use_container_width=True):
        # 隨機抽選一筆資料
        random_idx = np.random.randint(0, len(df))
        random_sample = df.iloc[random_idx]
        sample_X = X.iloc[random_idx].values.reshape(1, -1)
        
        # 載入對應模型
        model_mapping = {
            "KNN": "model_KNN.joblib",
            "LogisticRegression": "model_LogisticRegression.joblib",
            # "隨機森林": "model_RandomForest.joblib",
            "XGBoost": "model_XGBoost.joblib"
        }
        
        model_path = model_mapping[model_option]
        
        if Path(model_path).exists():
            model = joblib.load(model_path)
            
            # 進行預測
            prediction = model.predict(sample_X)[0]
            prediction_proba = model.predict_proba(sample_X)[0]
            
            # 儲存預測結果到 session state
            st.session_state.prediction = prediction
            st.session_state.prediction_proba = prediction_proba
            st.session_state.sample_X = sample_X
            st.session_state.random_sample = random_sample
            st.session_state.model_used = model_option
        else:
            st.error(f"找不到模型檔案: {model_path}")

# 顯示預測結果
if hasattr(st.session_state, 'prediction'):
    with col_predict2:
        st.subheader("預測結果")
        st.write(f"**使用模型**: {st.session_state.model_used}")
        st.write(f"**預測結果**: {'⚠️ 有違約風險' if st.session_state.prediction == 1 else '✅ 無違約風險'}")
        st.write(f"**預測概率分佈**:")
        prob_df = pd.DataFrame({
            "分類": ["無違約（0）", "違約（1）"],
            "概率": st.session_state.prediction_proba
        })
        st.bar_chart(prob_df.set_index("分類"))
        
        # 顯示抽選的資料樣本
        with st.expander("📊 檢視抽選的資料樣本"):
            st.write(st.session_state.random_sample)

