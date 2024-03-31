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
import time
from urllib.parse import quote

if 'docs' not in st.session_state:
    st.session_state.docs=None
if 'analyze' not in st.session_state:
    st.session_state.analyze=False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results=[]
if 'analysis_time' not in st.session_state:
    st.session_state.analysis_time=None
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs=[]

st.set_page_config(page_title='Black & Veatch | Information Security Contract Database and AI Analysis', layout='wide', page_icon='https://cdn.bfldr.com/E1EVDN8O/at/p3stnx8wsmbhx5p37f4sj89/23_BV_icon.eps?auto=webp&format=png')    
    
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

if 'contract_analysis_agent' not in st.session_state:
    llm=AzureChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"), 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment='gpt-4-32k',
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
        Only include the sections that you have referenced to come up with the therefore statement.
        i.e
        - **File X, Section x.x states:**
        - **File X, Section x.x states:**
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
def retrieve_chunks(query, docs, chunk_vector_store):
    filter=[]
    for doc in docs:
        filter.append(doc[0].metadata['filename'])
    chunks=chunk_vector_store.similarity_search_with_relevance_scores(query=query,k=2000,score_threshold=0.8)
    desired_chunks=[]
    for chunk in chunks:
        if chunk[0].metadata['filename'] in filter:
            chunk[0].page_content='File: '+chunk[0].metadata['filename']+', '+chunk[0].page_content
            desired_chunks.append(chunk[0])
    return(desired_chunks)




st.header(body='Black & Veatch | Information Security Contract Database and AI Analysis', divider='gray')
with st.container():
    vendor_name=st.chat_input(placeholder='Enter the name of a vendor')

if vendor_name!=None:
    st.session_state.vendor_name=vendor_name
    st.session_state.docs = st.session_state.full_doc_vector_store.similarity_search_with_relevance_scores(query=st.session_state.vendor_name,k=2000, score_threshold=0.83)
    
if st.session_state.docs!=None:
    with st.container(border=True):
        st.subheader('Search results for \"'+st.session_state.vendor_name+'\"')
        checkboxes=[]
        for doc in st.session_state.docs:
            doc_url=os.getenv('SHAREPOINT_FOLDER_URL')+quote(doc[0].metadata['filename'])
            checkboxes.append([st.checkbox(label='['+doc[0].metadata['filename']+']('+doc_url+')', value=True),doc[0].metadata['filename']])
    with st.container(border=True):
        st.subheader('AI Analysis Options:')
        st.session_state.analyze_validity=st.checkbox(label='Contract validity', value=True,help='AI performs analysis to determine the validity dates of the documents')
        st.session_state.analyze_notif_time=st.checkbox(label='When to notify', value=True, help='AI performs analysis to determine the notification period for cybersecurity incident')
        st.session_state.analyze_notif_contact=st.checkbox(label='Who to notify', value=True, help='AI performs analysis to determine who to contact if a cybersecurity incident occurs')
        st.session_state.analyze_report_info=st.checkbox(label='Information to report', value=True, help='AI performs analysis to determine what information must be reported if a cybersecurity incident occurs')
        st.session_state.analyze_data_reqs=st.checkbox(label='Data retention requirements', value=True, help='AI performs analysis to determine data retention requirements')
        st.session_state.analyze=st.button(label='Perform AI analysis', help='AI will perform analysis on the contracts for the vendor for selected options', use_container_width=True, type='primary')
    with st.container(border=True):
        if st.session_state.analyze:
            analysis_results=[]
            def analyze(question, docs, chunk_vector_store, contract_analysis_agent):
                global analysis_results
                chunks=retrieve_chunks(question, docs, chunk_vector_store)
                answer = contract_analysis_agent.invoke(input={'question':question,'input_documents':chunks})['output_text']
                analysis_results.append({'question':question, 'answer':answer})
            with st.spinner('Performing AI contract analysis...'):
                start=time.time()
                st.session_state.analysis_results=[]
                st.session_state.selected_docs=[]
                threads=[]
                for checkbox in checkboxes:
                    if checkbox[0]:
                        st.session_state.selected_docs.append(checkbox[1])
                if st.session_state.analyze_validity:
                    question='Timeframe when the contract is valid (start date to end date):'
                    threads.append(threading.Thread(target=analyze, args=(question, st.session_state.selected_docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                if st.session_state.analyze_notif_time:
                    question='Time to notify in the event of a cybersecurity incident:'
                    threads.append(threading.Thread(target=analyze, args=(question, st.session_state.selected_docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                if st.session_state.analyze_notif_contact:
                    question='Who to notify in the event of a cybersecurity incident, contact information (name/email/phone number and/or address):'
                    threads.append(threading.Thread(target=analyze, args=(question, st.session_state.selected_docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                if st.session_state.analyze_report_info:
                    question='Information to include when reporting the cybersecurity incident:'
                    threads.append(threading.Thread(target=analyze, args=(question, st.session_state.selected_docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))
                if st.session_state.analyze_data_reqs:
                    question='Data retention requirements:'
                    threads.append(threading.Thread(target=analyze, args=(question, st.session_state.selected_docs, st.session_state.chunk_vector_store, st.session_state.contract_analysis_agent), group=None))

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                st.session_state.analysis_results=analysis_results
                end=time.time()
                st.session_state.analysis_time=round(end-start,2)
        with st.container():
            if st.session_state.analysis_results!=[]:
                st.subheader('Analysis Results:')
                st.write('**Analysis completed in '+str(st.session_state.analysis_time)+' s**')
                st.write('*(Note: User is responsible for verifying all information)*')
                for result in st.session_state.analysis_results:
                    st.subheader(result['question'])
                    st.write(result['answer'])
        
                    

    
