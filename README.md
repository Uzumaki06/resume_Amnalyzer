So the task given was to extract relevant information from the resume and the same is done in this repository

# In order to run the code
- pip install -r requirements.txt
- set up all the resumes in the drive and create an API link for that drive so that it can be acessed easily.
- put in the xxxxxx.json file with the code so the retrieval process is completed without any authentication.
- Also add groq api key in .env file which has to be created differently.
- run the proces_resume file by using the following command -> streamlit run proces_resume.py

Used extract thinker library for extracting relevant information from the resumes as this library has shown great results while working with LLMs
Also modified the code in such a way that we have multiple models available so that if one model is not able to fulfil the needs the other can do it.This was done with the help of Lite LLM 
Allowed the code to process the resumes in batch size of 10 so that we can ensure batch_processing and allowed data vizualization so as to showcase all the results in a better way.
