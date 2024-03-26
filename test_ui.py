import streamlit as st
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
import os

if 'docs' not in st.session_state:
    st.session_state.docs=None
if 'chunk_vector_store' not in st.session_state:
    embedder = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", 
        chunk_size=1, 
        openai_api_key=os.getenv("OPENAI_API_KEY"), 
        openai_api_type=os.getenv("OPENAI_API_TYPE"), 
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    ) 

    st.session_state.chunk_vector_store= AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        embedding_function=embedder.embed_query,
        index_name=os.getenv('CHUNK_INDEX')
    )

    st.session_state.full_doc_vector_store = AzureSearch(
        azure_search_endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
        azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
        embedding_function=embedder.embed_query,
        index_name=os.getenv('FULL_DOC_INDEX')
    )

st.session_state.vendor_name=st.chat_input(placeholder='Enter the name of a vendor:')
with st.container():
    if st.session_state.vendor_name!=None:
        st.session_state.docs = st.session_state.full_doc_vector_store.similarity_search_with_relevance_scores(query=st.session_state.vendor_name,k=2000, score_threshold=0.83)
    
if st.session_state.docs!=None:
    st.write('The following documents are related to this vendor:')
    for doc in st.session_state.docs:
        st.write('- '+doc[0].metadata['filename'])

    
