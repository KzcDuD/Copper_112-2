import os
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import  SystemMessage,HumanMessage,AIMessage
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load API key from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"

# load json files
def load_jfile(file:list):
    folder_path = './data/'
    loader = JSONLoader(
        file_path=folder_path+file,
        jq_schema='.[]',
        content_key='html',
    )
    text_spilter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
    docs = text_spilter.split_documents(loader.load())
    return docs

class Chat_with_ai():
    def __init__(self,data=None,k=3):
        self.data = data
        self.query = None
        self.k = k
        self.messages = [
            SystemMessage(content="You are a helpful assistant for LangChain python framework."),
            SystemMessage(content="**Reply in zh-tw**"),
        ]
        # data ==none continue to chat
        self.db = 'None'
        self.transcript = 'None'
        if data != None:
            self.db = self.data_to_db(data)
            self.transcript = self.get()
    
    def data_to_db(self,data)->FAISS:
        embedding = OpenAIEmbeddings()
        db = FAISS.from_documents(data,embedding)
        return db
    
    def get(self):
        docs =self.db.similarity_search(query = self.query, k = self.k)
        # print(docs)
        docs_page_content = ' '.join([str(doc) for doc in docs])
        return docs_page_content
    
    def chat(self):
        model = ChatOpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model='gpt-3.5-turbo'
        )
        self.messages.append(SystemMessage(content=self.transcript))
        self.messages.append(HumanMessage(content=self.query))
        response =model.invoke(self.messages).content
        self.messages.append(AIMessage(content=response))
        return response

file = 'langchain_express.json'
j_data = load_jfile(file)
# pprint(j_data)
model = Chat_with_ai(j_data)

if __name__ == "__main__":
    # files =['langchain_intro.json','langchain_express.json']
    file = 'langchain_express.json'
    j_data = load_jfile(file)
    # pprint(j_data)
    query = 'what is LCEL?'
    model = Chat_with_ai(query,j_data)
    print(model.chat())


"""
NO data : LCEL 是 LangChain 框架的縮寫，代表 LangChain Execution Layer。這是 LangChain 框架的一部分，負責執行區塊鏈相關的操作和功能。


Have data: 

    LCEL（LangChain Expression Language）是一種聲明式的方式，可輕鬆地組合鏈接起來。LCEL從一開始就設計支援將原型放入生產環境，無需進行任何代碼更改，從最簡單的“提示+LLM”鏈到最複雜的鏈（我們已經看到人們成功在生產環境中運行帶有數百步驟的LCEL鏈）。以下是您可能想要使用LCEL的一些原因：

    - LCEL使得從基本組件構建複雜鏈變得容易。
    - 提供統一接口：每個LCEL對象都實現了Runnable接口，該接口定義了一組常用的調用方法（invoke、batch、stream、ainvoke等）。這使得LCEL對象的鏈也可以自動支持這些調用。換句話說，每個LCEL對象的鏈本身也是一個LCEL對象。
    - 組合原語：LCEL提供了一些原語，可輕鬆組合鏈、並行化組件、添加回退、動態配置鏈內部等。

    若想繼續學習有關LCEL的知識，建議您：

    - 閱讀完整的LCEL介面，我們在此處僅部分涵蓋了介面。
    - 探索如何部分以了解LCEL提供的其他組合原語。
    - 查看烹飪書部分，看看LCEL在常見用例中的應用。建議查看的下一個用例可能是檢索增強型生成。

    這些是瞭解LCEL的下一步。

"""