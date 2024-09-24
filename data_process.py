from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn.functional as F
import json
from tokenizers import ByteLevelBPETokenizer

import tensorflow as tf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('/content/drive/MyDrive/Colab Notebooks/drive-download-20231211T034526Z-001/vulnerables.json') as file:
  data = json.load(file)


# def filter_large_elements(data):
#     return [element for element in data if element['size'] <= 100]

# # Apply the filter function to the dataset
# filtered_dataset = filter_large_elements(data)
class ChunkandGroup(data):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files= '/content/drive/MyDrive/Colab Notebooks/drive-download-20231211T034526Z-001/vulnerables.json',
                    vocab_size=50257,
                    min_frequency=2,
                    special_tokens=["<s>",
                            "<pad>",
                            "</s>",
                            "<unk>",
                            "<mask>",
                            ] )

    tokenizer.save_model('./','tokenizer_1')
    tokenizer.save('./config.json')

    # Assuming you have already defined and trained the tokenizer as per your code


    tokenizer = RobertaTokenizer(vocab_file='./tokenizer_1-vocab.json',
                                merges_file='./tokenizer_1-merges.txt')


    code_snippets = [item['code'] for item in data]
    # print(code_snippets[2])
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    model.eval()
    # print(code_snippets)

    def chunk_text(text, chunk_size=510, pad_token_id=1):  # pad_token_id=1 for RoBERTa's <pad> token
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length = chunk_size)
        chunked = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Pad the last chunk if necessary
        if len(chunked) > 0 and len(chunked[-1]) < chunk_size:
            chunked[-1] += [pad_token_id] * (chunk_size - len(chunked[-1]))

        return chunked
    for i in range(len(code_snippets)):
        print(chunk_text(code_snippets[i]))
