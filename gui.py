import streamlit as st
import timeit
import utils

PATH_TO_DEFAULT_MODEL = utils.PATH_TO_DEFAULT_MODEL
PATH_TO_DEFAULT_VECTORDB = utils.PATH_TO_DEFAULT_VECTORDB

def create_page():
    """Create a frontend to be able to load model, select existing vector db of documents or folder with pdf
    Then ask question to the bot that will use th pdf to answer"""
    st.set_page_config(page_title="🤗💬 PdfChat")

    with st.sidebar:
        st.title('🤗💬 PdfChat')
        model = st.text_input('Enter path to model:', type='default')
        vector_db = st.text_input('Enter path to Pdf folder:', type='default')
        if st.button('Vectorise PDF') :
            if vector_db:
                #path_to_pdf = Path(vector_db)
                #print(f"Path : {path_to_pdf}")
                utils.embedded_pdf(vector_db)
                st.write('Pdf folder embed in db')
            else:
                st.write('Pdf folder path is missing')
        hf_pass = st.text_input('Enter vector DB:', type='default')

        if st.button('Launch ChatBot') :
            if model:
                path_to_model = model
                st.write('Use provided model')
            else :
                path_to_model = PATH_TO_DEFAULT_MODEL
                st.write('Defaut Model used')
            if vector_db :
                dbqa = utils.setup_QA(PATH_TO_DEFAULT_VECTORDB,path_to_model)
                st.session_state['dbqa'] = dbqa
                st.write('ChatBot launched')
            elif hf_pass :
                dbqa = utils.setup_QA(hf_pass,path_to_model)
                st.session_state['dbqa'] = dbqa
                st.write('ChatBot launched')
            else:
                st.write('You should provid folder with PDF or pdf already vectorized in a db')
    try:
        dbqa = st.session_state['dbqa']
    except:
        pass
    #Store answers
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # User-provided prompt
    if prompt := st.chat_input(disabled=not (vector_db or hf_pass)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = timeit.default_timer() # Start timer
                response = dbqa({'query':prompt} )
                end = timeit.default_timer() # End timer
                # Display time taken for CPU inference
                print(f"Time to retrieve response: {end - start}")
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == '__main__':

    create_page()
