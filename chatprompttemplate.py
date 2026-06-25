from langchain_core.prompts import ChatPromptTemplate


chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful {domain} expert'),
    ('human', 'Explain in simple terms what is {topic}')
])

prompts = chat_template.invoke({'domain': 'cricket', 'topic': 'cricket'})

print(prompts)