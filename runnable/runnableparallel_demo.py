from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
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
    template="Generate a post about {topic} in Instagram",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a post about {topic} in LinkedIN",
    input_variables=['topic']
)

chain = RunnableParallel({
    'Instagram': RunnableSequence(prompt1, model, parser),
    'LinkedIN': RunnableSequence(prompt2, model, parser)
})

results = chain.invoke({'topic': 'Cricket'})

print(results)