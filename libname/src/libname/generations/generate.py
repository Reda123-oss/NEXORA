import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


def generate(model , prompt , tokenizer , max_new_tokens : int , device = "cuda" ):
    tokens = tokenizer.encode(tokens) # output -> [id1 , id2 , id3 , ...]
    # Transform it to 2D array
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    start_pos = tokens.shape[1]

    for i in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)

        logits = logits[ : , -1 , :]
        probabilities = F.softmax(logits , dim = -1)
        next_token = torch.multinomial(probabilities , 1)
        tokens = torch.cat([tokens , next_token] , dim = -1)
    generated_text = tokenizer.decode(tokens[0].to_list())
    return generated_text