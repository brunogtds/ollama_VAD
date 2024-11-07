# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:25:36 2024

@author: bruno
"""

import ollama

response = ollama.chat(model='llama3.2:latest', messages=[
  {
    'role': 'user',
    'content': 'Who was Luke Skywalker?',
  },
])
print(response['message']['content'])


import pandas as pd

df= pd.read_csv("C:/Users/bruno/Downloads/EmoBank-v1.4/JULIELab-EmoBank-3ca77cf/corpus/emobank.csv")
df.head()

import re
import csv 

prompt = f"Determine a value of valence (in the range 1.00 - 9.00), arousal (in the range 1.00 - 9.00) and dominance (in the range 1.00 - 9.00) considering the VAD model for Sentiment Analysis, for the following sentence.  Do not explain your decision, only answer the values for valence, arousal and dominance. Your output must be in the format 'Valence: <value> Arousal: <value> Dominance: <value>'."

def get_response (prompt, passagem):
    response = ollama.chat(model='llama3.2:latest', messages=[
  {
    'role': 'user',
    'content': f'{prompt}\nEXCERPT: "{passagem}\n"'
  },
])
    content = response['message']['content'].strip()
    return content

output= get_response(prompt, "Remember what she said in my last letter?")
#response = output['message']['content'].strip()
print(output)

def get_vad_scores(text):
    prompt = f"Determine a value of valence (in the range 1.00 - 9.00), arousal (in the range 1.00 - 9.00) and dominance (in the range 1.00 - 9.00) considering the VAD model for Sentiment Analysis, for this sentence: '{text}'.  Do not explain your decision, only answer the values for valence, arousal and dominance. Your output must be in the format 'Valence: <value> Arousal: <value> Dominance: <value>'."
    
    
    output = get_response(prompt, text)
    #response = output['message']['content'].strip()
    #print(response)
    print(output)
    vad_scores = {}
    try:
        vad_scores['Valence'] = float(re.search(r"Valence:\s*([\d.]+)", output).group(1))
        vad_scores['Arousal'] = float(re.search(r"Arousal:\s*([\d.]+)", output).group(1))
        vad_scores['Dominance'] = float(re.search(r"Dominance:\s*([\d.]+)", output).group(1))
    except (IndexError, ValueError):
        # Handle any extraction errors gracefully, e.g., by setting default values
        vad_scores = {'Valence': None, 'Arousal': None, 'Dominance': None}
        print(output)

    return vad_scores

vad_results = []
df= df.head(100)

for idx, row in df.iterrows():
    text = row['text']
    vad_scores = get_vad_scores(text)
    vad_results.append([row['id'], row['split'], vad_scores['Valence'], vad_scores['Arousal'], vad_scores['Dominance']])
    
# Create a new DataFrame with the results
vad_results_df = pd.DataFrame(vad_results, columns=['id', 'split', 'Ollama_Valence', 'Ollama_Arousal', 'Ollama_Dominance'])

# Combine with original dataset for easy comparison
combined_df = pd.merge(df, vad_results_df, on=['id', 'split'], how='left')
combined_df.to_csv('C:/Users/bruno/Downloads/output.csv', index=False)
