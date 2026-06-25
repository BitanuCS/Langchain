from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

class Review(TypedDict):
    summary: Annotated[list[str],"Generate summary"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative or positive "]

structured_output = model.with_structured_output(Review)

results = structured_output.invoke("""Efficient Cooling & Trusted Brand – Great Purchase!

I recently purchased the Voltas Air Conditioner (by Tata) from Amazon, and I am extremely satisfied with the product. The cooling is quick and effective, even during peak summer temperatures.

The installation was smooth, and the unit blends well with the room interiors. I also appreciate the build quality and sleek design. Voltas, being a trusted Tata brand, lives up to its reputation for reliability and customer service.

Overall, I would highly recommend this product for anyone looking for a dependable and efficient AC at a reasonable price. Great value for money!
                         """)

print(results['summary'])