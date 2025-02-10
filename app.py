#!/usr/bin/env python
# coding: utf-8

# In[4]:


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
import os
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.vectorstores import Chroma 
import gradio as gr 
from langchain_groq import ChatGroq


# In[16]:


def initialize_llm():
    llm=ChatGroq(temperature=0, groq_api_key="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",model_name="llama-3.3-70b-versatile")
    return llm

def create_vector_db():
    loader=DirectoryLoader("data",glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db=Chroma.from_documents(texts,embeddings,persist_directory='./chroma_db')
    vector_db.persist()

    print("ChromaDB created and data saved")

    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # ✅ Retrieve up to 3 relevant results

    # ✅ Properly using memory in the QA system
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Enables structured chat history
    )

    # ✅ Improved and simplified prompt
    prompt_template = """
    You are **Aryabhatta**, an astronomy chatbot created by **Anwesa**.  
    Your goal is to provide **clear, engaging, and informative** answers about space science.  
    
    🌌 **Important Notes**:  
    - If asked about your creator, **acknowledge Anwesa as your creator**.  
    - If no relevant context is found, answer based on general knowledge.  
    - Be **concise yet informative** and adjust complexity based on user familiarity.  
    - If asked, include **relevant links** to scientific articles, research papers, or reputable astronomy websites.  
    - **Recall past user preferences** (e.g., their interests, favorite topics, and prior discussions) and answer accordingly.  

    ---  
    🔭 **Relevant Context**: {context}  
    🪐 **User Question**: {question}  
    
    🚀 **Aryabhatta Responds:**  
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=['question', 'context'])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}  # ✅ Fix applied here
    )

    return qa_chain  



print("Initializing Chatbot......") 
llm = initialize_llm()  # ✅ Ensure this function exists
db_path = "chroma_db/"


if not os.path.exists(db_path):  
    vector_db = create_vector_db()  # ✅ Create DB if it doesn't exist
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)  # ✅ Ensure this function exists

# while True:
#     query = input("\nHuman: ")
#     if query.lower() == "exit":
#         print("Aryabhatta: Take Care Brotha, Tada!")
#         break

  
#     # ✅ Ensure proper dictionary structure for LangChain
#     response = qa_chain({"question": query, "chat_history": []})

#     # ✅ Extract the "answer" field correctly
#     if 'answer' in response:
#         print(f"Aryabhatta: {response['answer']}")
#     else:
#         print("Aryabhatta: Sorry, I couldn't process that.")



# if __name__ == "__main__": 
#      main()


def chatbot_response(user_input, history):
    if not user_input.strip():
        return "Please provide a valid input"
    
    response = qa_chain({"question": user_input, "chat_history": history})
    answer = response.get("answer", "I'm not sure about that, but I'd love to learn more with you! 🚀")

    return answer  # ✅ No manual history append

with gr.Blocks(theme='NoCrypt/miku@1.0.0') as app:
    chatbot=gr.ChatInterface(fn=chatbot_response,title="Arybhatta, Your Astronomy Buddy!")

app.launch()

