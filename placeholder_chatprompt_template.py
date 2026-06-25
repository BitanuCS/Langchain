from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate([
    ('system','You are an intelligent assistant'),
    MessagesPlaceholder(variable_name = 'chat_history')
    ('human','query')
])

chat_history = []

with open('history.txt') as f:
    chat_history.extend(f.readline())
    

print(chat_history)

results = template.invoke({'chat_history':chat_history, 'query':'Where is my refund?'})

print(results.content)