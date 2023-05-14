import openai
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding
import time
import credentials
import tiktoken
import pinecone

# openai api key
openai.api_key = credentials.OPENAI_API_KEY

# pinecone api key
pinecone.init(api_key=credentials.PINECONE_API_KEY, environment=credentials.PINECONE_ENV_KEY)

#webcraping content from website
URL = "https://rohjag18.medium.com/convolutional-neural-networks-dont-memorize-learn-instead-a4fbf3604a54"
req = requests.get(URL)
soup = BeautifulSoup(req.content, 'html.parser')
paragraph = ""

#creating empty dataframe
user_info_df = pd.DataFrame(columns=['User_data'])

#filtering content to get purely text wanted
articles = soup.findAll("p", attrs={"class": "pw-post-body-paragraph"})
paragraph = articles[0].get_text() # only getting text from webscraping

# for article in articles:
#     # getting data into dataframe
#     tempDf = pd.DataFrame(article, columns=['User_data'])
#     user_info_df = pd.concat([user_info_df, tempDf], ignore_index=True)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


print(paragraph)
print(num_tokens_from_string("tiktoken is great!", "cl100k_base"))
#training embeddings w delay cuz rate limit
# def train_embeddings(text):
#     new_embeddings = get_embedding(text, engine='text-embedding-ada-002')
#     user_info_df['embeddings'].add(new_embeddings)
#     time.sleep(0.5)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

user_info_df['User_data'] = get_embedding(paragraph, model="text-embedding-ada-002")

print(user_info_df["User_data"])

# user_info_df['User_data'] = user_info_df.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)

# #getting the vector values for current dataframe
# user_info_df['embeddings'] = user_info_df['User_data'].apply(lambda x: train_embeddings(x))

# print(get_embedding(paragraph, engine='text-embedding-ada-002'))
# print(user_info_df['User_data'])
# print(user_info_df)