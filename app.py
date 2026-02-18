import streamlit as st
import pandas as pd
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import os

# --- 設定頁面資訊 ---
st.set_page_config(page_title="海扶及達文西問答小幫手", page_icon="🤖")
st.title("海扶及達文西問答小幫手 🤖")
st.markdown("輸入問題，即可獲得專業回覆！如果仍有疑問，可透過 Line 進一步諮詢。")

# --- 設定 Google Gemini API ---
# 嘗試從 Streamlit Secrets 讀取 (部署時使用)，若無則嘗試環境變數，或讓使用者在側邊欄輸入
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("請輸入 Google API Key", type="password")

if not api_key:
    st.info("請輸入 Google API Key 以啟動機器人。")
    st.stop()

genai.configure(api_key=api_key)

# --- 定義 Gemini Embedding Function 給 ChromaDB 使用 ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/text-embedding-004"
        # 為了效能，這裡逐筆呼叫 (Gemini API 支援 batch 但需視限額調整)
        embeddings = []
        for text in input:
            response = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_query"
            )
            embeddings.append(response['embedding'])
        return embeddings

# --- 初始化與載入資料庫 (使用快取避免重複載入) ---
@st.cache_resource
def initialize_vector_db():
    # 建立 ChromaDB 客戶端 (使用記憶體模式或短暫儲存，適合 Streamlit Cloud)
    chroma_client = chromadb.Client() 
    
    # 檢查是否已有 collection，若無則建立
    try:
        collection = chroma_client.get_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
    except ValueError:
        collection = chroma_client.create_collection(
            name="medical_faq",
            embedding_function=GeminiEmbeddingFunction()
        )
        
        # 讀取 Excel 資料
        try:
            # 假設 Excel 檔案與 app.py 在同一目錄
            data = pd.read_excel("網路問答.xlsx")
            
            # 確保欄位名稱正確，防止錯誤
            if '問題' in data.columns and '回覆' in data.columns:
                questions = data['問題'].astype(str).tolist()
                answers = data['回覆'].astype(str).tolist()
                
                ids = [f"id-{i}" for i in range(len(questions))]
                
                # 寫入 ChromaDB
                collection.add(
                    documents=answers,  # 搜尋結果回傳的是答案 (Document)
                    metadatas=[{"question": q} for q in questions],
                    ids=ids
                )
            else:
                st.error("Excel 檔案格式錯誤：找不到 '問題' 或 '回覆' 欄位。")
        except FileNotFoundError:
            st.error("找不到 '網路問答.xlsx' 檔案，請確認已上傳。")
            
    return collection

# 載入資料庫
collection = initialize_vector_db()

# --- 初始化聊天歷史紀錄 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 顯示歷史訊息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- 處理使用者輸入 ---
if prompt := st.chat_input("請輸入您的醫療問題..."):
    # 1. 顯示使用者訊息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 搜尋最相似的問答
    results = collection.query(
        query_texts=[prompt],
        n_results=1
    )

    # 取得距離與答案 (ChromaDB 預設是 L2 距離，越小越相似)
    # 註：Gemini Embedding 的距離判斷標準可能與 OpenAI 不同，建議根據測試調整閾值
    # 這裡暫設 0.6，若發現常回答不出，可調高此數值
    distance = results['distances'][0][0] if results['distances'] else 1.0
    best_answer = results['documents'][0][0] if results['documents'] else ""

    # 3. 判斷邏輯
    final_response = ""
    
    # 設定信心門檻 (數字越小代表越相似)
    # 注意：Chroma 預設 L2 distance，OpenAI 的 0.5 約對應 L2 的 0.5-0.7 左右，需微調
    THRESHOLD = 0.6 

    if distance > THRESHOLD:
        # 信心不足，回傳預設罐頭訊息 (支援 HTML)
        final_response = (
            "這個問題比較複雜，建議您至門診進一步諮詢醫師。<br><br>"
            "<b>門診時間：</b><br>"
            "- 林口長庚醫院：週二上午、週六下午<br>"
            "- 土城醫院：週二下午、週六上午<br><br>"
            "如果您有更多疑問，也歡迎透過 "
            "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold;'>Line 小編</a> "
            "進一步線上諮詢哦！"
        )
    else:
        # 信心足夠，呼叫 Gemini 生成溫暖回覆
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            system_prompt = f"""
            你是陳醫師的專業且親切的醫療諮詢助理。
            使用者的問題是：{prompt}
            資料庫檢索到的標準答案是：{best_answer}
            
            請根據標準答案，用溫暖、自然且口語化的方式回答使用者。
            回答請保持簡潔有力，不要長篇大論。
            不要編造資料庫中沒有的醫學事實。
            """
            
            response = model.generate_content(system_prompt)
            gpt_reply = response.text
            
            # 加上 Line 連結 footer
            final_response = gpt_reply + (
                "<br><br>如果您有更多疑問，也歡迎透過 "
                "<a href='https://line.me/R/ti/p/@hifudr' target='_blank' style='color: #4CAF50; font-weight: bold;'>Line 小編</a> "
                "進一步線上諮詢哦！"
            )
            
        except Exception as e:
            final_response = f"抱歉，系統暫時繁忙，請稍後再試。(錯誤代碼: {e})"

    # 4. 顯示並儲存助手回覆
    with st.chat_message("assistant"):
        st.markdown(final_response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
