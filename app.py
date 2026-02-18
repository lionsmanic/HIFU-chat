import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os
import time

# ==========================================
# 1. 頁面設定與金鑰讀取
# ==========================================
st.set_page_config(
    page_title="海扶及達文西問答小幫手",
    page_icon="🤖",
    layout="centered"
)

st.title("海扶及達文西問答小幫手 🤖")
st.markdown("輸入問題，即可獲得專業回覆！")

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。")
    st.stop()

# ==========================================
# 2. 定義 Gemini Embedding (優化版：不掃描，直接試)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # 優先嘗試新版 Embedding，失敗則退回舊版
        model_candidates = ["models/text-embedding-004", "models/embedding-001"]
        
        embeddings = []
        for text in input:
            success = False
            for model_name in model_candidates:
                try:
                    response = genai.embed_content(
                        model=model_name,
                        content=text,
                        task_type="retrieval_query"
                    )
                    embeddings.append(response['embedding'])
                    success = True
                    break # 成功就跳出迴圈
                except Exception:
                    continue # 失敗就試下一個模型
            
            if not success:
                # 真的都失敗，回傳空向量避免當機
                embeddings.append([0.0]*768)
                
        return embeddings

# ==========================================
# 3. 初始化資料庫 (加入快取與進度提示)
# ==========================================
@st.cache_resource(show_spinner="正在載入醫療知識庫...")
def initialize_vector_db():
    client = chromadb.Client()
    
    # 嘗試讀取或建立資料庫
    collection = client.get_or_create_collection(
        name="medical_faq",
        embedding_function=GeminiEmbeddingFunction()
    )
    
    # 若資料庫是空的，從 Excel 載入
    if collection.count() == 0:
        excel_file = "網路問答.xlsx"
        if os.path.exists(excel_file):
            try:
                data = pd.read_excel(excel_file)
                if '問題' in data.columns and '回覆' in data.columns:
                    data = data.dropna(subset=['問題', '回覆'])
                    questions = data['問題'].astype(str).tolist()
                    answers = data['回覆'].astype(str).tolist()
                    ids = [f"id-{i}" for i in range(len(questions))]
                    
                    collection.add(
                        documents=answers,
                        metadatas=[{"question": q} for q in questions],
                        ids=ids
                    )
                    print(f"✅ 資料庫建立完成，共 {len(questions)} 筆。")
                else:
                    st.error("❌ Excel 格式錯誤。")
            except Exception as e:
                st.error(f"❌ 讀取 Excel 失敗: {e}")
    return collection

# 執行初始化
try:
    collection = initialize_vector_db()
except Exception as e:
    st.error(f"資料庫初始化失敗: {e}")
    st.stop()

# ==========================================
# 4. 聊天邏輯 (加入視覺回饋)
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入您的醫療問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    final_response = ""
    
    # === 步驟 1: 搜尋資料庫 ===
    with st.spinner('🔍 正在搜尋相關醫療資訊...'):
        try:
            results = collection.query(
                query_texts=[prompt],
                n_results=1
            )
            distance = results['distances'][0][0] if results['distances'] else 1.0
            best_answer = results['documents'][0][0] if results['documents'] else ""
        except Exception as e:
            st.error(f"搜尋失敗: {e}")
            distance = 1.0 # 強制視為找不到

    # === 步驟 2: 判斷與生成 ===
    THRESHOLD = 0.65

    if distance > THRESHOLD:
        final_response = (
            "這個問題比較複雜，建議您至門診進一步諮詢醫師。<br><br>"
            "<b>🏥 門診時間：</b><br>"
            "- 林口長庚醫院：週二上午、週六下午<br>"
            "- 土城醫院：週二下午、週六上午<br><br>"
            "歡迎透過 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 小編</a> 線上諮詢。"
        )
    else:
        # === 步驟 3: AI 生成回答 (顯示轉圈圈) ===
        with st.spinner('🤖 AI 正在整理回答...'):
            try:
                # 定義要嘗試的模型清單 (優先 -> 備用)
                chat_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
                
                generated_text = ""
                model_used = ""
                
                # 快速嘗試模型
                for model_name in chat_models:
                    try:
                        chat_model = genai.GenerativeModel(model_name)
                        system_prompt = f"""
                        你是一位專業且溫暖的婦科醫療諮詢助理，隸屬於陳威君醫師團隊。
                        使用者的問題是：{prompt}
                        資料庫檢索到的標準答案是：{best_answer}
                        請根據標準答案，用溫暖、自然且口語化的方式回答。
                        """
                        response = chat_model.generate_content(system_prompt)
                        generated_text = response.text
                        model_used = model_name
                        break # 成功就跳出
                    except Exception:
                        continue # 失敗換下一個
                
                if generated_text:
                    final_response = generated_text + (
                        "<br><br>---<br>"
                        "如有疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>。"
                    )
                else:
                    final_response = "抱歉，系統目前繁忙，請稍後再試。"
                    
            except Exception as e:
                final_response = f"系統發生錯誤: {str(e)}"

    # 顯示結果
    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
