from fastapi import FastAPI
from langserve import add_routes
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints._common import NVEModel
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader
from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda,RunnableBranch
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from functools import partial
from operator import itemgetter
from functools import partial
from getpass import getpass
from keras.models import load_model
import numpy as np
import gradio as gr
import requests
import asyncio
import uvicorn
import PyPDF2
import os

########################################################################
## load the embedded documents
embedder = NVIDIAEmbeddings(model="nvolveqa_40k")
docstore = FAISS.load_local("docstore_index", embedder,allow_dangerous_deserialization=True)
docs = list(docstore.docstore._dict.values())

## Make some custom Chunks to give big-picture details
doc_string = ""
doc_metadata = []
for doc in docs:
    metadata = doc.metadata
    if (metadata.get('Title')!= None) and (metadata.get('Title') not in doc_string):
        doc_string += "\n - " + metadata.get('Title')
        doc_metadata += [str(metadata)]


########################################################################
## Utility Runnables/Methods
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)
embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string."""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## Reorders longer documents to center of output text
long_reorder = RunnableLambda(LongContextReorder().transform_documents)
########################################################################

embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")
llm = ChatNVIDIA(model="mixtral_8x7b") | StrOutputParser()
convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

initial_msg = (
    "Hello! I am a document chat agent here to help the user!"
    f" I have access to the following documents: {doc_string}\n\nHow can I help you?"
)

model1 = load_model("filter.h5")
def is_good_response(query):
    # embed the query and pass the embedding into your classifier
    embedding = np.array([embedder.embed_query(query)])
    # return true if it's most likely a good response and false otherwise
    return model1(embedding)

good_sys_msg = (
    "You are an NVIDIA chatbot. Please answer their question if it is ethical and relevant while representing NVIDIA."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Only cite sources that are used. Make your response conversational.)"
)
## Resist talking about this topic" system message
poor_sys_msg = (
    "You are an NVIDIA chatbot. Please answer their question while representing NVIDIA."
    "  Their question has been analyzed and labeled as 'probably not useful to answer as an NVIDIA Chatbot',"
    "  so avoid answering if appropriate and explain your reasoning to them. Make your response as short as possible."
)

chat_prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("user", "{input}")])

retrieval_chain = (
    {'input' : (lambda x: x)}
    | RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
    | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
    | RPrint()
)

stream_chain = (
    { 'input'  : (lambda x:x), 'is_good' : is_good_response }
    | RPrint()
    | RunnableBranch(
            # bad question
            ((lambda d: d['is_good'] < 0.5), RunnableAssign(dict(system = RunnableLambda(lambda x: poor_sys_msg))) | chat_prompt | llm),
            # good question
            RunnableAssign(dict(system = RunnableLambda(lambda x: good_sys_msg)))| RunnableAssign({'history' : itemgetter('input') | convstore.as_retriever() | long_reorder | docs2str})
                | RunnableAssign({'context' : itemgetter('input') | docstore.as_retriever()  | long_reorder | docs2str})
                | RPrint() |chat_prompt | llm
    )
)


def chat_gen(message, history=[], return_buffer=True):
#     print(type(message))
#     print("message:\n")
#     print(message)
    buffer = ""
    line_buffer = ""
    ## load the uploaded pdf into the existing vecstore
    if (len(message['files'])>0) and (".pdf" in message['files'][0]):
        with open(message['files'][0], 'rb') as file:
            loader = PyPDFLoader(message['files'][0]).load()
            print("Adding new document into vector database...")
            chunks = [text_splitter.split_documents(loader)]
            vecstore = [FAISS.from_documents(chunk, embedder) for chunk in chunks]
            for vstore in vecstore:
                docstore.merge_from(vstore)
        ## if 
        if "Title" in loader[0].metadata:
            buffer+="I have received your document '"+loader[0].metadata.Title+"'. I'm glad to help if you have any question regarding it. "
            yield buffer
        else:
            first_line = loader[0].page_content.split('\n')[0]
            buffer+="I have received your document '"+first_line+"'. I'm glad to help if you have any question regarding it. "
            yield buffer

    ## response to the user input message
    if len(message['text'].strip()) > 0:
    
        ## Then, stream the results of the stream_chain
        for token in stream_chain.stream(message['text']):
            buffer += token
            ## keep line from getting too long
            if not return_buffer:
                line_buffer += token
                if "\n" in line_buffer:
                    line_buffer = ""
                if ((len(line_buffer)>84 and token and token[0] == " ") or len(line_buffer)>100):
                    line_buffer = ""
                    yield "\n"
                    token = "  " + token.lstrip()
            yield buffer if return_buffer else token

    elif len(message['files'])==0:
        buffer+="Please do not send whitespaces. "
        yield buffer
    
    ## Lastly, save the chat exchange to the conversation memory buffer
    save_memory_and_get_output({'input':  message['text'], 'output': buffer}, convstore)

chatbot = gr.Chatbot(value = [[None, initial_msg]],height=720)
demo = gr.ChatInterface(chat_gen, chatbot=chatbot,multimodal=True ).queue()

try:
    demo.launch(debug=True, share=True, show_api=False)
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e

"""
The script can be adapted to provide apis as a backend service following the codes below.(tested on Windows)
We choose to directly build frontend based on gradio.
"""
# import nest_asyncio
# nest_asyncio.apply()
# async def run_backend():
#     app = FastAPI(
#         title="LangChain Server",
#         version="1.0",
#         description="A simple api server using Langchain's Runnable interfaces",
#     )

#     add_routes(app, llm, path="/basic_chat")
#     add_routes(app, stream_chain, path="/rag_chat")

#     uvicorn_config = uvicorn.Config(app, host="0.0.0.0", port=9012)
#     server = uvicorn.Server(uvicorn_config)
#     await server.serve()

# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(run_backend())
