import streamlit as st
import google.generativeai as genai
import importlib.metadata
import os

st.set_page_config(page_title="系統診斷工具", page_icon="🛠️")
st.title("🛠️ AI 系統診斷報告")

st.info("此工具用於檢測 API Key 權限與套件版本，請根據下方紅字修正問題。")

# ==========================================
# 檢查 1: 套件版本
# ==========================================
st.subheader("1. 檢查 Python 套件版本")
try:
    pkg_name = "google-generativeai"
    ver = importlib.metadata.version(pkg_name)
    
    # 檢查是否大於 0.8.0
    is_new_enough = tuple(map(int, ver.split('.')[:3])) >= (0, 8, 0)
    
    if is_new_enough:
        st.success(f"✅ {pkg_name} 版本: {ver} (符合需求)")
    else:
        st.error(f"❌ {pkg_name} 版本過舊: {ver}")
        st.warning("👉 請修改 requirements.txt 為 `google-generativeai>=0.8.3` 並點擊 Reboot app。")
except Exception as e:
    st.error(f"❌ 無法偵測套件: {e}")

# ==========================================
# 檢查 2: API Key 格式與權限
# ==========================================
st.subheader("2. 檢查 API Key 連線")

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ 尚未設定 Secrets 變數 `GOOGLE_API_KEY`")
    st.stop()
else:
    # 檢查是否有常見錯誤（如多餘空白、引號）
    if api_key.strip() != api_key:
        st.warning("⚠️ 警告：您的 API Key 前後包含空白，系統已自動修復。請檢查 Secrets 設定。")
    if api_key.startswith('"') or api_key.startswith("'"):
        st.error("❌ 錯誤：您的 API Key 不應該包含引號。請在 Secrets 中移除引號。")
    
    st.success(f"✅ API Key 已讀取 (開頭: {api_key[:5]}...)")
    genai.configure(api_key=api_key)

# ==========================================
# 檢查 3: 實際列出可用模型 (關鍵！)
# ==========================================
st.subheader("3. 帳號權限與模型清單")

try:
    st.write("正在連線 Google 伺服器取得模型清單...")
    models = list(genai.list_models())
    
    model_names = [m.name for m in models]
    
    # 顯示 raw list 供參考
    with st.expander("點此查看完整模型清單"):
        st.json(model_names)

    # 檢查關鍵模型是否存在
    target_chat = "models/gemini-1.5-flash"
    target_embed = "models/text-embedding-004"
    
    if target_chat in model_names:
        st.success(f"✅ 您的帳號支援最新模型: {target_chat}")
    else:
        st.error(f"❌ 您的帳號找不到 {target_chat}")
        st.info("👉 這代表您的 API Key 可能是在 Google Cloud Console 申請的 (Vertex AI)，或者該 Key 沒有權限。請至 [Google AI Studio](https://aistudio.google.com/) 重新申請 Key。")

    if target_embed in model_names:
        st.success(f"✅ 您的帳號支援嵌入模型: {target_embed}")
    else:
        st.error(f"❌ 您的帳號找不到 {target_embed}")

    # ==========================================
    # 檢查 4: 實際生成測試
    # ==========================================
    st.subheader("4. 實際生成測試")
    
    if target_chat in model_names:
        with st.spinner("正在測試 Gemini 1.5 Flash 回應..."):
            try:
                model = genai.GenerativeModel(target_chat)
                response = model.generate_content("Hello, reply 'OK' if you see this.")
                st.success(f"✅ 生成成功！AI 回應: {response.text}")
            except Exception as e:
                st.error(f"❌ 生成失敗 (權限不足或額度已滿): {e}")
    else:
        st.warning("跳過生成測試 (因找不到模型)")

except Exception as e:
    st.error(f"❌ 連線嚴重錯誤: {e}")
    st.markdown("""
    **可能原因：**
    1. API Key 無效。
    2. 您的 Streamlit 伺服器 IP 被 Google 封鎖 (較少見)。
    3. 您使用的不是 AI Studio Key。
    """)
