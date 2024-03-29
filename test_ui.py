import streamlit as st
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import StuffDocumentsChain
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import threading

if 'docs' not in st.session_state:
    st.session_state.docs=None
if 'analyze' not in st.session_state:
    st.session_state.analyze=False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results=[]
if 'update_agent' not in st.session_state:
    st.session_state.update_agent=True

def update_agent():
    st.session_state.update_agent=True

st.set_page_config(page_title='Black & Veatch | Information Security Contract Database and AI Analysis', layout='wide')    
st.session_state.gpt_version=st.sidebar.selectbox(label='GPT Version', options=['gpt-35-turbo-16k','gpt-4'], index=0, help='GPT LLM version used to perform contract analysis', placeholder='gpt-35-turbo', on_change=update_agent)
    
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

if 'contract_analysis_agent' not in st.session_state or st.session_state.update_agent==True:
    llm=AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=st.session_state.gpt_version,
        streaming=True,
        temperature=0.4
    )
    doc_summ_template=(
        """
        SYSTEM: You are in an intelligent assistant which analyzes portions of contracts that Black & Veatch (an engineering, procurement and construction company has with its vendors and clients.
        The contracts are related to information security/cyber security.
        Extract the required information from the legal contracts, including paragraph or section numbers and filenames
        Think step by step.
        Be brief and straight to the point in answering the required information.
        If there are no legal contracts provided, state: 'The selected documents do not contain the requested information.'
        
    
        Provide your answer in bullet points:
        i.e
        - File X, Section x.x states:
        - File X, Section x.x states:
        - **Therefore:**
    
        Required Information: {question}
        Legal Contracts: {context}
    
        """
    )
    docsumm_prompt=PromptTemplate.from_template(doc_summ_template)
    docsumm_llm=LLMChain(llm=llm,prompt=docsumm_prompt)
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "question"],
        template="{page_content}"
    )
    
    st.session_state.contract_analysis_agent=StuffDocumentsChain(
        llm_chain=docsumm_llm,
        document_prompt=document_prompt,
        document_variable_name='context',
        verbose=True
    )
    
    st.session_state.update_agent=False
def retrieve_chunks(query):
    filter=[]
    for doc in st.session_state.docs:
        filter.append(doc[0].metadata['filename'])
    chunks=st.session_state.chunk_vector_store.similarity_search_with_relevance_scores(query=query,k=2000,score_threshold=0.8)
    desired_chunks=[]
    for chunk in chunks:
        if chunk[0].metadata['filename'] in filter:
            chunk[0].page_content='File: '+chunk[0].metadata['filename']+', '+chunk[0].page_content
            desired_chunks.append(chunk[0])
    return(desired_chunks)




st.header(body='Black & Veatch | Information Security Contract Database and AI Analysis', divider='gray')
with st.container():
    st.session_state.vendor_name=st.chat_input(placeholder='Enter the name of a vendor:')

if st.session_state.vendor_name!=None:
    st.session_state.docs = st.session_state.full_doc_vector_store.similarity_search_with_relevance_scores(query=st.session_state.vendor_name,k=2000, score_threshold=0.83)
    
if st.session_state.docs!=None:
    with st.container():
        st.write('The following documents are related to this vendor:')
        with st.container():
            for doc in st.session_state.docs:
                st.checkbox(label=doc[0].metadata['filename'], value=True)
        with st.container():
            st.write('AI Analysis Options:')
            st.session_state.analyze_validity=st.checkbox(label='Contract validity', value=True,help='AI performs analysis to determine the validity dates of the documents')
            st.session_state.analyze_notif_time=st.checkbox(label='When to notify', value=True, help='AI performs analysis to determine the notification period for cybersecurity incident')
            st.session_state.analyze_notif_contact=st.checkbox(label='Who to notify', value=True, help='AI performs analysis to determine who to contact if a cybersecurity incident occurs')
            st.session_state.analyze_report_info=st.checkbox(label='Information to report', value=True, help='AI performs analysis to determine what information must be reported if a cybersecurity incident occurs')
            st.session_state.analyze_data_reqs=st.checkbox(label='Data retention requirements', value=True, help='AI performs analysis to determine data retention requirements')
            st.session_state.analyze=st.button(label='Perform AI analysis', help='AI will perform analysis on the contracts for the vendor for selected options', use_container_width=True)
        with st.container():
            if st.session_state.analyze:
                def analyze(question, docs, chunk_vector_store, contract_analysis_agent):
                    st.session_state.docs=docs
                    st.session_state.chunk_vector_store=chunk_vector_store
                    st.session_state.contract_analysis_agent=contract_analysis_agent
                    chunks=retrieve_chunks(question)
                    answer = st.session_state.contract_analysis_agent.invoke(input={'question':question,'input_documents':chunks})['output_text']
                    st.session_state.analysis_results.append({'question':question, 'answer':answer})
                with st.spinner('Performing AI contract analysis...'):
                    st.session_state.analysis_results=[]
                    threads=[]
                    if st.session_state.analyze_validity:
                        question='Timeframe when the contract is valid (start date to end date):'
                        threads.append(threading.Thread(target=analyze, args=(question, st.session_state.docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                    if st.session_state.analyze_notif_time:
                        question='Time to notify in the event of a cybersecurity incident:'
                        threads.append(threading.Thread(target=analyze, args=(question, st.session_state.docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                    if st.session_state.analyze_notif_contact:
                        question='Who to notify in the event of a cybersecurity incident, contact information (name/email/phone number and/or address):'
                        threads.append(threading.Thread(target=analyze, args=(question, st.session_state.docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                    if st.session_state.analyze_report_info:
                        question='Information to include when reporting the cybersecurity incident:'
                        threads.append(threading.Thread(target=analyze, args=(question, st.session_state.docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))

                    for thread in threads:
                        thread.start()
                        thread.join()
            with st.container():
                if st.session_state.analysis_results!=[]:
                    for result in st.session_state.analysis_results:
                        st.subheader(result['question'])
                        st.write(result['answer'])
            
                    

    
