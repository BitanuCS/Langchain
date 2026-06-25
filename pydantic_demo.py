from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

class Review(BaseModel):
    # summary: str = Field (description="Generate summary")
    # sentiment: Literal["pos", "neg"] = Field(description = "Return sentiment of the review either negative or positive ")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    

# structured_output = model.with_structured_output(Review)

# results = structured_output.invoke("""Efficient Cooling & Trusted Brand – Great Purchase!

# I recently purchased the Voltas Air Conditioner (by Tata) from Amazon, and I am extremely satisfied with the product. The cooling is quick and effective, even during peak summer temperatures.

# The installation was smooth, and the unit blends well with the room interiors. I also appreciate the build quality and sleek design. Voltas, being a trusted Tata brand, lives up to its reputation for reliability and customer service.

# Overall, I would highly recommend this product for anyone looking for a dependable and efficient AC at a reasonable price. Great value for money!
# """)

# print(results)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(result)