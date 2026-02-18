import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# ==========================================
# 1. 頁面與除錯設定
# ==========================================
st.set_page_config(page_title="海扶醫療問答 (診斷版)", page_icon="🩺")
st.title("海扶醫療問答 🩺")

# 側邊欄：系統健康狀態
st.sidebar.header("🔍 系統診斷資訊")

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    st.sidebar.success("API Key 已讀取")
else:
    st.error("❌ 尚未設定 Google API Key。")
    st.stop()

# ==========================================
# 2. 核心：絕對動態模型選擇器 (抓到什麼用什麼)
# ==========================================
@st.cache_resource
def get_available_models():
    """
    強制的模型偵測：
    1. 列出所有模型。
    2. 不管名稱叫什麼，只要支援 generateContent 就拿來當聊天模型。
    3. 只要支援 embedContent 就拿來當嵌入模型。
    """
    chat_models = []
    embed_models = []
    
    try:
        # 嘗試列出所有模型
        for m in genai.list_models():
            # 判斷是否支援對話
            if 'generateContent' in m.supported_generation_methods:
                chat_models.append(m.name)
            # 判斷是否支援嵌入
            if 'embedContent' in m.supported_generation_methods:
                embed_models.append(m.name)
        
        return chat_models, embed_models
    except Exception as e:
        st.sidebar.error(f"無法列出模型清單: {e}")
        return [], []

# 執行偵測
ALL_CHAT_MODELS, ALL_EMBED_MODELS = get_available_models()

# 顯示偵測結果在側邊欄 (讓使用者知道發生什麼事)
st.sidebar.write("---")
st.sidebar.subheader("可用的聊天模型：")
st.sidebar.json(ALL_CHAT_MODELS)
st.sidebar.subheader("可用的嵌入模型：")
st.sidebar.json(ALL_EMBED_MODELS)

# 決策邏輯：優先順序
def select_best_model(model_list, priority_keywords):
    if not model_list:
        return None
    
    # 嘗試尋找優先關鍵字
    for keyword in priority_keywords:
        for model in model_list:
            if keyword in model:
                return model
    
    # 如果都沒對中，直接回傳第一個能用的 (絕不回傳假字串)
    return model_list[0]

# 選定模型
FINAL_CHAT_MODEL = select_best_model(ALL_CHAT_MODELS, ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"])
FINAL_EMBED_MODEL = select_best_model(ALL_EMBED_MODELS, ["text-embedding-004", "embedding-001"])

if not FINAL_CHAT_MODEL:
    st.error("❌ 嚴重錯誤：您的 API Key 無法存取任何聊天模型。請檢查 Google AI Studio 的 API 權限。")
    st.stop()

if not FINAL_EMBED_MODEL:
    st.error("❌ 嚴重錯誤：您的 API Key 無法存取任何嵌入模型。")
    st.stop()

st.sidebar.write("---")
st.sidebar.success(f"✅ 最終選用聊天模型: {FINAL_CHAT_MODEL}")
st.sidebar.success(f"✅ 最終選用嵌入模型: {FINAL_EMBED_MODEL}")


# ==========================================
# 3. 定義 Embedding (使用選定的模型)
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=FINAL_EMBED_MODEL, # 絕對使用偵測到的模型
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                # 若失敗，印出錯誤但不崩潰
                print(f"Embedding error: {e}")
                embeddings.append([0.0]*768)
        return embeddings

# ==========================================
# 4. 初始化資料庫
# ==========================================
@st.cache_resource(show_spinner="正在載入資料庫...")
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
                    st.error("Excel 格式錯誤。")
            except Exception as e:
                st.error(f"讀取 Excel 失敗: {e}")
    return collection

try:
    collection = initialize_vector_db()
except Exception as e:
    st.error(f"資料庫錯誤: {e}")
    st.stop()

# ==========================================
# 5. 聊天邏輯
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("請輸入您的醫療問題..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
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
            with st.spinner(f'🤖 AI ({FINAL_CHAT_MODEL}) 思考中...'):
                # 使用偵測到的模型
                model = genai.GenerativeModel(FINAL_CHAT_MODEL)
                
                system_prompt = f"""
                你是專業的醫療助理。
                使用者問題：{prompt}
                標準答案：{best_answer}
                請根據標準答案親切回答。
                """
                
                response = model.generate_content(system_prompt)
                final_response = response.text + (
                    "<br><br>---<br>"
                    "如有疑問，歡迎 <a href='https://line.me/R/ti/p/@hifudr' target='_blank'>Line 線上諮詢</a>。"
                )

    except Exception as e:
        final_response = f"系統錯誤: {e}"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
