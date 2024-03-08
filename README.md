# jen-document-summarization

In this lab, we will build a simple document summarizer with Amazon Bedrock, LangChain, and Streamlit.

LangChain includes a map-reduce summarization function that allows us to process content that exceeds the token limit for the model. The map-reduce function works by breaking up a document into smaller pieces, summarize those pieces, then summarize the pieces' summaries.

You can build the application code by copying the code snippets below and pasting into the indicated Python file.

This lab might fail if it exceeds your throttling rate limit.

Just want to run the app?
You can jump ahead to run a pre-made application.

Use cases
The map-reduce summarization pattern is good for the following use cases:

- Summarizing long documents
- Summarizing call transcripts
- Summarizing customer activity history
- Architecture
- Architecture diagram, illustrating the introduction above

The Map-Reduce pattern involves the following steps:

1. Break up the large document into small chunks
2. Generate intermediate summaries based on those small chunks
3. Summarize the intermediate summaries into a combined summary

This application consists of two files: one for the Streamlit front end, and one for the supporting library to make calls to Bedrock.

 
**Create the library script**
First we will create the supporting library to connect the Streamlit front end to the Bedrock back end.

Navigate to the workshop/labs/summarization folder, and open the file summarization_lib.py

Add the import statements.

These statements allow us to use LangChain to load a PDF file, split the document, and call Bedrock.

You can use the copy button in the box below to automatically copy its code:

import os
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


 

Add a function to create a Bedrock LangChain client.

This includes the inference parameters we want to use.
def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 8000, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


 

Add the function to create the document chunks.

This code attempts to split the document by paragraph, line, sentence, or word.
pdf_path = "2022-Shareholder-Letter.pdf"

def get_docs():
    
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], chunk_size=4000, chunk_overlap=100 
    )
    docs = text_splitter.split_documents(documents=documents)
    
    return docs


 

Add this function to call Bedrock.

This code creates prompts for the map and reduce steps. It then passes the documents to the map-reduce summarizer chain to produce the combined summary.
def get_summary(return_intermediate_steps=False):
    
    map_prompt_template = "{text}\n\nWrite a few sentences summarizing the above:"
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    combine_prompt_template = "{text}\n\nWrite a detailed analysis of the above:"
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    
    
    llm = get_llm()
    docs = get_docs()
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, return_intermediate_steps=return_intermediate_steps)
    
    if return_intermediate_steps:
        return chain({"input_documents": docs}, return_only_outputs=True) #make return structure consistent with chain.run(docs)
    else:
        return chain.run(docs)


 

Save the file.
Excellent! You are done with the backing library. Now we will create the front-end application.

 

**Create the Streamlit front-end app** 

In the same folder as your lib file, open the file summarization_app.py
 

Add the import statements.

These statements allow us to use Streamlit elements and call functions in the backing library script.
import streamlit as st
import summarization_lib as glib


 

Add the page title and configuration.

Here we are setting the page title on the actual page and the title shown in the browser tab.
st.set_page_config(layout="wide", page_title="Document Summarization")
st.title("Document Summarization")


 

Add the summarization elements.

This section will not be displayed until the session state's has_document property has been set.
We are creating a checkbox and button to get the user's prompt and send it to Bedrock. The "Return intermediate steps" checkbox determines if we will display the summaries from the map stage.
We use the if block below to handle the button click. We display a spinner while the backing function is called, then write the output to the web page.

return_intermediate_steps = st.checkbox("Return intermediate steps", value=True)
summarize_button = st.button("Summarize", type="primary")


if summarize_button:
    st.subheader("Combined summary")

    with st.spinner("Running..."):
        response_content = glib.get_summary(return_intermediate_steps=return_intermediate_steps)


    if return_intermediate_steps:

        st.write(response_content["output_text"])

        st.subheader("Section summaries")

        for step in response_content["intermediate_steps"]:
            st.write(step)
            st.markdown("---")

    else:
        st.write(response_content)


 

Save the file.
Superb! Now you are ready to run the application!

 

**Run the Streamlit app** 

Select the bash terminal in AWS Cloud9 and change directory.
cd ~/environment/workshop/labs/summarization

Just want to run the app?
Expand here & run this command instead
 

Run the streamlit command from the terminal.
streamlit run summarization_app.py --server.port 8080

Ignore the Network URL and External URL links displayed by the Streamlit command. Instead, we will use AWS Cloud9's preview feature.

 

In AWS Cloud9, select Preview -> Preview Running Application.
Screenshot of terminal, showing the Cloud9 preview button

You should see a web page like below:

Streamlit app at launch

 

Click the summarize button.

The summarization may take about 90 seconds to run.
Once complete, the combined summary for the document will be displayed, along with the individual section summaries if you chose to return the intermediate steps.
This lab might fail if it exceeds your throttling rate limit.

Streamlit app in use

 

Close the preview tab in AWS Cloud9. Return to the terminal and press Control-C to exit the application.
