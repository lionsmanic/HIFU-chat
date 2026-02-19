# ==========================================
# 0. 系統環境修正 (SQLite Fix) - 必放第一行
# ==========================================
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 介面設定
# ==========================================
st.set_page_config(page_title="海扶醫療諮詢室 (除錯版)", page_icon="🛠️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #fcfcfc; font-family: "Microsoft JhengHei", sans-serif; }
    h1 { color: #d32f2f; font-weight: 700; border-bottom: 2px solid #e0e0e0; padding-bottom: 15px; }
    [data-testid="stSidebar"] {display: none;}
    .stChatMessage { border-radius: 15px; border: 1px solid #f0f0f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ 海扶醫療諮詢 (除錯模式)")
st.info("⚠️ 目前為除錯模式，若發生錯誤將會顯示完整代碼。")

# ==========================================
# 2. API Key 讀取與測試
# ==========================================
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 系統錯誤：未設定 API Key。")
    st.stop()

# ==========================================
# 3. 資料庫邏輯
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = genai.embed_content(model="models/text-embedding-004", content=input, task_type="retrieval_query")
            return [response['embedding']] if 'embedding' in response else [e for e in response['embedding']]
        except Exception as e:
            # 這裡不隱藏錯誤，直接讓它爆出來，這樣我們才知道 Embedding 壞了
            print(f"Embedding Error: {e}")
            # 備用
            try:
                res = genai.embed_content(model="models/embedding-001", content=input)
                return [res['embedding']]
            except:
                return [[0.0]*768 for _ in input]

@st.cache_resource(show_spinner="正在載入資料庫...")
def initialize_vector_db():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(name="medical_faq", embedding_function=GeminiEmbeddingFunction())
        
        if collection.count() == 0:
            excel_file = "網路問答.xlsx"
            if os.path.exists(excel_file):
                data = pd.read_excel(excel_file).dropna(subset=['問題', '回覆'])
                questions = data['問題'].astype(str).tolist()
                answers = data['回覆'].astype(str).tolist()
                ids = [f"id-{i}" for i in range(len(questions))]
                
                # 這裡不分批了，直接寫入看看會不會爆
                collection.add(documents=answers, metadatas=[{"question": q} for q in questions], ids=ids)
        return collection
    except Exception as e:
        st.error(f"❌ 資料庫崩潰: {str(e)}")
        return None

collection = initialize_vector_db()

# ==========================================
# 4. 對話邏輯 (顯示真實錯誤)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "我是除錯機器人，請輸入問題，我會測試連線。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入測試問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if collection is None:
        st.error("資料庫未啟動，無法測試。")
        st.stop()
    
    final_response = ""
    
    with st.spinner('🤖 正在暴力測試連線...'):
        try:
            # 1. 搜尋
            try:
                results = collection.query(query_texts=[prompt], n_results=1)
                best_answer = results['documents'][0][0] if results['documents'] else ""
                st.write(f"✅ 資料庫搜尋成功，找到答案：{best_answer[:20]}...")
            except Exception as e:
                st.error(f"❌ 資料庫搜尋失敗: {e}")
                st.stop()

            # 2. AI 生成 (不隱藏錯誤！)
            candidates = ["gemini-1.5-flash", "gemini-pro"]
            success = False
            error_log = []
            
            for model_name in candidates:
                try:
                    # 測試生成
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(f"請回答：{prompt}")
                    final_response = response.text
                    success = True
                    st.toast(f"✅ 模型 {model_name} 成功！")
                    break
                except Exception as e:
                    error_msg = str(e)
                    error_log.append(f"{model_name}: {error_msg}")
                    # 繼續試下一個
            
            if not success:
                # 這裡會顯示最真實的錯誤訊息
                final_response = f"❌ 所有模型連線失敗！<br><b>錯誤詳情：</b><br>" + "<br>".join(error_log)

        except Exception as e:
            final_response = f"❌ 系統發生預期外錯誤: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
