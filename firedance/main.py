import boto3.session
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain.schema.runnable import RunnableSequence
import os
import boto3
import streamlit as st

os.environ["AWS_REGION"] = ""

bedrock_client = boto3.Session(profile_name="",
    service_name = "bedrock-runtime",
    region_name = os.getenv(),
)

modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"

llm = ChatBedrock(
       client=bedrock_client,
       model_id=modelId,
       model_kwargs={"max_retries": 3},
   )

def my_chatbot(language: str, freeform_text:str) -> str:
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a helpful assistant. Translate the following text into {language}: {freeform_text}",
    )

    # 여러 컴포넌트를 순차적으로 실행할 수 있도록 리스트로 받음
    components = [prompt, llm]
    bedrock_chain = RunnableSequence(prompt, llm)
    response = bedrock_chain.invoke({"language": language, "freeform_text": freeform_text})
    return response

if __name__ == "__main__":
    st.title("Language Translation Chatbot")
    freeform_text = st.text_area("Enter text to translate")

    language = st.text_input("Enter target language (e.g., Korean, French, etc.)")
    if st.button("Translate"):
        if freeform_text and language:
            response = my_chatbot(language, freeform_text)
            st.write("Translation:", response)
        else:
            st.warning("Please enter both the text to translate and the target language.")
