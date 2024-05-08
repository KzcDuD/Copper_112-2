from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import  SystemMessage,HumanMessage,AIMessage
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS


from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or "YOUR_API_KEY"

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
        #  self.db = self.data_to_db(data)
        # self.transcript = self.get()
    
    # def data_to_db(self,data)->FAISS:
    #     embeddings = OllamaEmbeddings()
    #     query_result = embeddings.embed_query(data)
    #     db = FAISS.from_documents(data,embeddings)
    #     return db
    
    # def get(self):
    #     docs = self.db.similarity_search(query = self.query, k = self.k)
    #     # print(docs)
    #     docs_page_content = ' '.join([doc for doc in docs])
    #     return docs_page_content
    
    def chat(self):
        groq = ChatGroq(temperature=0, model_name="llama2-70b-4096")
        # self.messages.append(SystemMessage(content=self.transcript))
        prompt = ChatPromptTemplate.from_messages([
            ("system","You are a helpful assistant for LangChain python framework."),
            ("system","**Reply in zh-tw**"),
            ('human',self.query)
            ]
        )
        chain = prompt | groq
        for chunk in chain.stream({"Query": self.query}):
            print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    # files =['langchain_intro.json','langchain_express.json']
    file = 'langchain_express.json'
    j_data = load_jfile(file)
    # pprint(j_data)
    query = 'what is LCEL?'
    model = Chat_with_ai(j_data,query)
    print(model.chat())