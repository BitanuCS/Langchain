from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template = "Give a brief discurssion about the topic {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template = "Give a 5 line summary about the text {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

results = chain.invoke({'topic': 'blackhole'})

print(results)