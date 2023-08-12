# Question you PDF

Use llama 2 to question your PDF through a Streamlit front end.  

You should download a llama GGML model in order to be able to provide it to the front end.  

You should define a path to store the vector db of your pdf file through PATH_TO_DEFAULT_VECTORDB in utils.py. 

First time, provide the folder with the PDF you need to embed.  
Launch vectorisation, the db will be store in the path define earlier.  

Next time you can indicate the path to the vector DB and launch the bot.  

GGML model could be download on https://huggingface.co/TheBloke.  
For using Apple m1 acceleration from llama_cpp_python model should be a q4_0 quantization. Model name should end by 'q4_0.bin'. 

## References
https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/. 
https://huggingface.co/TheBloke. 
https://github.com/kennethleungty/Llama-2-Open-Source-LLM-CPU-Inference/tree/main. 
