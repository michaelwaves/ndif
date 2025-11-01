import torch
from nnsight import LanguageModel, CONFIG
import pandas as pd
import os
CONFIG.API.HOST = 'api.ndif.us'
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

df = pd.read_csv('https://raw.githubusercontent.com/saprmarks/geometry-of-truth/refs/heads/main/datasets/cities.csv')
df = df.iloc[:100]
df.head()
model = LanguageModel("meta-llama/Llama-2-7b-hf", device_map="auto")

def rindex(lst, value):
  """get the rightmost index of a value in a list."""
  return len(lst) - 1 - lst[::-1].index(value)
PROMPT_TEMPLATE = """The city of Tokyo is in Japan. This statement is: TRUE
The city of Hanoi is in Poland. This statement is: FALSE
{statement} This statement is:"""

LAYER = 20
punctuation_token_id = model.tokenizer('.').input_ids[1] # extract activation over "."

true_activations = []
false_activations = []
with model.session(remote=True) as session:
  for i in range(df.shape[0]):
    row = df.iloc[i]
    prompt = PROMPT_TEMPLATE.format(statement=row.statement)
    prompt_token_ids = model.tokenizer(prompt).input_ids
    punctuation_index = rindex(prompt_token_ids, punctuation_token_id) - 1

    with model.trace(prompt):
      activation = model.model.layers[LAYER].output[0][:, punctuation_index, :].save()
    if row.label == 0:
      false_activations.append(activation)
    else:
      true_activations.append(activation)


true_activations = torch.cat(true_activations)
false_activations = torch.cat(false_activations)

true_activations.shape, false_activations.shape