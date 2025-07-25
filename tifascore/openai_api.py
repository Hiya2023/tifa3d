import os
from openai import OpenAI
import time, sys

api_key=os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
client=OpenAI(api_key=api_key)

#removed the max_tokens
#max_tokens=800

#changing to gpt-4o which superceded gpt 3.5 turbo
def openai_completion(prompt, engine="gpt-4o", temperature=0):
    print("openai api call")
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        #max_tokens=max_tokens,
        temperature=temperature,
        #stop=["\n\n", "<|endoftext|>"]
        )
    
    return resp.choices[0].message.content



