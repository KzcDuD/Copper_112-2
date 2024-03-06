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
    def __init__(self,data,query,k=3):
        self.data = data
        self.query = query
        self.k = k
        self.messages = [
            SystemMessage(content="You are a helpful assistant for LangChain python framework."),
            SystemMessage(content="**Reply in zh-tw**"),
        ]
        self.db = self.data_to_db(data)
        self.transcript = self.get()
    
    def data_to_db(self,data)->FAISS:
        embedding = OpenAIEmbeddings()
        db = FAISS.from_documents(data,embedding)
        return db
    
    def get(self):
        docs =self.db.similarity_search(query = self.query, k = self.k)
        # print(docs)
        docs_page_content = ' '.join([doc for doc in docs])
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


if __name__ == "__main__":
    # files =['langchain_intro.json','langchain_express.json']
    file = 'langchain_express.json'
    j_data = load_jfile(file)
    # pprint(j_data)
    query = 'what is LCEL?'
    model = Chat_with_ai(j_data,query)
    print(model.transcript)
