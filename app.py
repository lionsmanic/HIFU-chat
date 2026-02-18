import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 頁面設定與金鑰讀取
# ==========================================
st.set_page_config(
    page_title="海扶及達文西問答小幫手 (除錯模式)",
    page_icon="🛠️",
    layout="centered"
)

st.title("海扶及達文西問答小幫手 🛠️")
st.warning("目前為除錯模式：若發生錯誤，將會顯示詳細代碼以供排查。")

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。")
    st.stop()

# ==========================================
# 2. 定義 Gemini Embedding (除錯版)
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
                    break 
                except Exception as e:
                    # 在後台印出錯誤，但不中斷前端
                    print(f"Embedding {model_name} error: {e}")
                    continue 
            
            if not success:
                embeddings.append([0.0]*768)
                
        return embeddings

# ==========================================
# 3. 初始化資料庫
# ==========================================
@st.cache_resource(show_spinner="正在載入醫療知識庫...")
def initialize_vector_db():
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="medical_faq",
        embedding_function=GeminiEmbeddingFunction()
    )
    
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
                else:
                    st.error("❌ Excel 格式錯誤。")
            except Exception as e:
                st.error(f"❌ 讀取 Excel 失敗: {e}")
    return collection

try:
    collection = initialize_vector_db()
except Exception as e:
    st.error(f"資料庫初始化失敗: {e}")
    st.stop()

# ==========================================
# 4. 聊天邏輯 (顯示詳細錯誤)
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
    
    # 搜尋資料庫
    results = collection.query(query_texts=[prompt], n_results=1)
    distance = results['distances'][0][0] if results['distances'] else 1.0
    best_answer = results['documents'][0][0] if results['documents'] else ""

    THRESHOLD = 0.65

    if distance > THRESHOLD:
        final_response = (
            "這個問題比較複雜，建議您至門診進一步諮詢醫師。<br><br>"
            "<b>🏥 門診時間：</b><br>"
            "- 林口長庚醫院：週二上午、週六下午<br>"
            "- 土城醫院：週二下午、週六上午"
        )
    else:
        with st.spinner('🤖 AI 正在連線中...'):
            # === 除錯重點區 ===
            # 直接嘗試呼叫，若失敗則把錯誤印出來給您看
            try:
                # 嘗試建立模型 (這裡加上 models/ 前綴比較保險)
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                
                system_prompt = f"""
                你是專業的醫療助理。
                使用者問題：{prompt}
                標準答案：{best_answer}
                請根據標準答案親切回答。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + "<br><br>(回應來源: Gemini-1.5-Flash)"
                
            except Exception as e1:
                # 第一個失敗，試試看舊版 Pro
                st.error(f"⚠️ Gemini 1.5 Flash 呼叫失敗: {e1}")
                st.info("🔄 嘗試切換至 Gemini 1.0 Pro...")
                
                try:
                    model = genai.GenerativeModel("models/gemini-pro")
                    response = model.generate_content(system_prompt)
                    final_response = response.text + "<br><br>(回應來源: Gemini-1.0-Pro)"
                except Exception as e2:
                    # 如果都失敗，顯示紅字錯誤
                    st.error(f"❌ 所有模型皆失敗。")
                    st.code(f"錯誤 1 (Flash): {e1}\n錯誤 2 (Pro): {e2}", language="text")
                    final_response = "系統連線錯誤，請截圖上方的錯誤訊息給管理員。"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
