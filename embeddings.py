import openai
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding


# openai api key
openai.api_key = "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" 

#webcraping content from website
URL = "https://rohjag18.medium.com/convolutional-neural-networks-dont-memorize-learn-instead-a4fbf3604a54"
req = requests.get(URL)
soup = BeautifulSoup(req.content, 'html.parser')

#creating empty dataframe
user_info_df = pd.DataFrame(columns=['User_data'])

#filtering content to get purely text wanted
articles = soup.findAll("p", attrs={"class": "pw-post-body-paragraph"})
for article in articles:
    # getting data into dataframe
    tempDf = pd.DataFrame(article, columns=['User_data'])
    user_info_df = pd.concat([user_info_df, tempDf], ignore_index=True)


#getting the vector values for current dataframe
user_info_df['embeddings'] = user_info_df['User_data'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
print(user_info_df)