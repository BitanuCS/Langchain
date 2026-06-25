from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Create a joke about this topic \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke \n {text}",
    input_variables=['text']
)

chain1 = RunnableSequence(prompt1, model, parser)

results1 = chain1.invoke({'topic': 'Cricket'})



chain2 = RunnableSequence(prompt2, model, parser)

results2 = chain2.invoke({'text': results1})

print(results1,"\n")
print(results2)