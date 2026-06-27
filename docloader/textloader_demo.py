from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

loader = TextLoader("docloader/cricket.txt",encoding="utf-8")

docs = loader.load()

print(docs[0])

prompt = PromptTemplate(
    template="Write Summary of the following text \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser

results = chain.invoke({'text': docs[0].page_content})

print(results)
