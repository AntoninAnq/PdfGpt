from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import cpuinfo


PATH_TO_DEFAULT_VECTORDB = ''
PATH_TO_DEFAULT_MODEL = ''

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def embedded_pdf(path_to_pdf):
    """Create vector db from a folder containing PDF files"""

    # Load PDF files from folder
    loader = DirectoryLoader(path_to_pdf,
                        glob="*.pdf",
                        loader_cls=PyPDFLoader)

    print(f"Load PDF")
    document = loader.load()

    # Split text from PDF into chunks
    print("Launch splitting")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
    texts = text_splitter.split_documents(document)

    # Load embeddings model
    print("Launch embedding")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

    # Create vector db
    print("Store data")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(PATH_TO_DEFAULT_VECTORDB)

def test_cpu_arch():
    """Test if python is running in m1 mac for using llama_cpp"""
    manufacturer = cpuinfo.get_cpu_info().get('brand_raw')
    arch = 'arm' if 'm1' in manufacturer.lower() else 'x86_64'
    return arch

def setup_model(path_to_model):
    """Create Ctransformer object by loading LLM model"""
    llm = CTransformers(model=path_to_model, # Location of downloaded GGML model
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
    return llm


def setup_GPU_model(path_to_model):
    """Create LLM model using appleM1 mps GPU acceleration"""
    n_gpu_layers = 1
    n_batch = 1024
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
    model_path=path_to_model,
    input={"temperature": 0.05, "max_length": 2000, "top_p": 1},
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
    )
    return llm


def set_qa_prompt():
    """Use the defined template to create a PromptTemplate object"""
    prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(qa_template),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    )
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    """RetrievalQA for questioning the bot"""
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={"verbose": True, 'prompt': prompt,
                                                          "memory": ConversationBufferMemory(memory_key="history",
                                                                                             input_key="question",
                                                                                             return_messages=True)})
    return dbqa


def setup_dbqa(llm,path_to_db):
    """Concat the whole config to create chatbot"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(path_to_db, embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
    return dbqa


def setup_QA(path_to_db,path_to_model):
    """Launch chatbot"""
    print(f"model end : {path_to_model[-8:]}")
    if test_cpu_arch() == 'arm' and path_to_model[-8:] == 'q4_0.bin':
        llm = setup_GPU_model(path_to_model)
    elif test_cpu_arch() == 'x86_64':
        llm = setup_model(path_to_model)

    dbqa = setup_dbqa(llm,path_to_db)
    return dbqa
