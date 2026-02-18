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
    page_title="海扶及達文西問答小幫手",
    page_icon="🤖",
    layout="centered"
)

st.title("海扶及達文西問答小幫手 🤖")
st.markdown("輸入問題，即可獲得專業回覆！如果仍有疑問，可透過 Line 進一步諮詢。")

# 讀取 API Key
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 尚未設定 Google API Key。請在 Secrets 中設定 'GOOGLE_API_KEY'。")
    st.stop()

# ==========================================
# 🚀 2. 動態模型偵測函數 (核心修改)
# ==========================================
@st.cache_resource
def get_best_chat_model_name():
    """動態偵測並回傳最佳可用的聊天模型名稱"""
    try:
        # 列出所有可用模型
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # 定義優先順序 (越前面越優先)
        priority_list = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro",
            "models/gemini-pro"
        ]
        
        # 1. 先找優先清單中的模型
        for priority in priority_list:
            # 比對時忽略 models/ 前綴差異，確保相容
            for available in available_models:
                if priority in available or available in priority:
                    print(f"✅ 自動選用聊天模型: {available}")
                    return available
        
        # 2. 如果都沒有，就回傳第一個找到的可生成模型
        if available_models:
            print(f"⚠️ 無法使用優先模型，降級使用: {available_models[0]}")
            return available_models[0]
            
        return "gemini-pro" # 最後的保底
    except Exception as e:
        print(f"❌ 模型偵測失敗: {e}")
        return "gemini-pro"

@st.cache_resource
def get_embedding_model_name():
    """動態偵測並回傳最佳可用的 Embedding 模型"""
    try:
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                if 'text-embedding-004' in m.name:
                    return m.name
        return "models/text-embedding-004" # 預設
    except:
        return "models/embedding-001" # 舊版備用

# 取得偵測到的模型名稱
CHAT_MODEL_NAME = get_best_chat_model_name()
EMBEDDING_MODEL_NAME = get_embedding_model_name()

# ==========================================
# 3. 定義 Gemini Embedding 函數
# ==========================================
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model=EMBEDDING_MODEL_NAME, # 使用動態偵測到的名稱
                    content=text,
                    task_type="retrieval_query"
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Embedding error: {e}")
                embeddings.append([0.0]*768)
        return embeddings

# ==========================================
# 4. 初始化資料庫
# ==========================================
@st.cache_resource
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
                    print(f"✅ 成功載入 {len(questions)} 筆問答資料。")
                else:
                    st.error("❌ Excel 檔案格式錯誤：找不到 '問題' 或 '回覆' 欄位。")
            except Exception as e:
                st.error(f"❌ 讀取 Excel 失敗: {e}")
        else:
            st.warning(f"⚠️ 找不到 '{excel_file}'，請確認檔案已上傳至 GitHub。")
            
    return collection

collection = initialize_vector_db()

# ==========================================
# 5. 聊天視窗邏輯
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
        results = collection.query(
            query_texts=[prompt],
            n_results=1
        )

        distance = results['distances'][0][0] if results['distances'] else 1.0
        best_answer = results['documents'][0][0] if results['documents'] else ""

        THRESHOLD = 0.65 

        if distance > THRESHOLD:
            final_response = (
                "這個問題比較複雜，建議您至門診進一步諮詢醫師，以獲得最準確的評估。<br><br>"
                "<b>🏥 門診時間：</b><br>"
                "- 林口長庚醫院：週二上午、週六下午<br>"
                "- 土城醫院：週二下午、週六上午<br><br>"
                "如果您有更多疑問，也歡迎透過 "
                "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold; text-decoration: none;'>Line 小編</a> "
                "進一步線上諮詢哦！"
            )
        else:
            # 使用動態偵測到的模型名稱
            chat_model = genai.GenerativeModel(CHAT_MODEL_NAME)
            
            system_prompt = f"""
            你是一位專業且溫暖的婦科醫療諮詢助理，隸屬於陳威君醫師團隊。
            
            【任務目標】
            使用者的問題是：{prompt}
            資料庫檢索到的標準答案是：{best_answer}
            
            請根據「標準答案」的內容，用更口語、親切、像真人的方式回答使用者。
            
            【回答準則】
            1. 語氣要溫柔、有同理心，讓患者感到安心。
            2. 內容必須準確基於標準答案，不可自行編造醫學事實。
            3. 結尾可以適當加上鼓勵的話。
            4. 保持簡潔。
            """
            
            response = chat_model.generate_content(system_prompt)
            final_response = response.text + (
                "<br><br>---<br>"
                "如果您有更多疑問，歡迎透過 "
                "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold; text-decoration: none;'>Line 小編</a> "
                "線上諮詢。"
            )

    except Exception as e:
        # 如果還是失敗，顯示更詳細的錯誤資訊幫助除錯
        final_response = f"抱歉，系統暫時繁忙 (使用模型: {CHAT_MODEL_NAME})。錯誤訊息: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
