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
from tqdm.auto import tqdm
from time import sleep
import helper

# openai api key
openai.api_key = credentials.OPENAI_API_KEY
print(openai.Engine.list) # checks authentication
embed_model = "text-embedding-ada-002" # specifying model

# pinecone api key
pinecone.init(api_key=credentials.PINECONE_API_KEY, environment=credentials.PINECONE_ENV_KEY)
index_name = "second-conscience"

# complete function
def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

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
#training embeddings w delay cuz rate limit
# def train_embeddings(text):
#     new_embeddings = get_embedding(text, engine='text-embedding-ada-002')
#     user_info_df['embeddings'].add(new_embeddings)
#     time.sleep(0.5)

# getting embeddings
res = openai.Embedding.create(
    input=paragraph, 
    engine=embed_model
)
# checking the records for embeddings
print(len(res["data"]))

# check if pinecone index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())

new_data = helper.token(paragraph.split(" "))

batch_size = 100 #how many embeddings we create and input at once
for i in tqdm(range(0, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)


# searching for query vector xq
query = "How do computers see?"
res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=2, include_metadata=True)

limit = 3750
prompt = ""
def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt


query_with_contexts = retrieve(query)
print(complete(query_with_contexts))



# def get_embedding(text, model="text-embedding-ada-002"):
#    text = text.replace("\n", " ")
#    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# user_info_df['User_data'] = get_embedding(paragraph, model="text-embedding-ada-002")

# print(user_info_df["User_data"])

# user_info_df['User_data'] = user_info_df.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)

# #getting the vector values for current dataframe
# user_info_df['embeddings'] = user_info_df['User_data'].apply(lambda x: train_embeddings(x))

# print(get_embedding(paragraph, engine='text-embedding-ada-002'))
# print(user_info_df['User_data'])
# print(user_info_df)