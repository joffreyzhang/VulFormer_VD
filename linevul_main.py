from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import joblib
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
from linevul_model import BertModel
from positional_encoding import PositionalEncoding, PositionalEncoding2D
from edge_encoding import CPGProcessor
from type_encoding import TypeEmbedding
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, PrecisionRecallDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
from tokenizers import Tokenizer
import json

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 connection_matrix):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.connection_matrix = connection_matrix

        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        super(TextDataset, self).__init__()
        self.cpg_directory = args.cpg_directory
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_json(file_path)
        funcs = df["func"].tolist()
        labels = df["target"].tolist()
        for i in tqdm(range(len(funcs))):
            if funcs[i].count('\n') < 100:
                cpg_path = os.path.join(self.cpg_directory, f"{i}.c", "export.json")
                if os.path.exists(cpg_path):
                    with open(cpg_path, 'r') as file:
                        cpg_data = json.load(file)
                    self.examples.append(convert_examples_to_features(funcs[i], labels[i], cpg_data, tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),torch.tensor(self.examples[i].connection_matrix)
# def preprocess_source_code(source_code):
#     # Regular expressions for single-line and multi-line comments
#     single_line_comment_re = r"//.*"
#     multi_line_comment_re = r"/\*[\s\S]*?\*/"

#     # Function to add special tokens around comments
#     def add_special_tokens(match):
#         return f"<comment> {match.group()} </comment>"

#     # Add special tokens to multi-line comments
#     source_code = re.sub(multi_line_comment_re, add_special_tokens, source_code)
#     # Add special tokens to single-line comments
#     source_code = re.sub(single_line_comment_re, add_special_tokens, source_code)

#     # Add <code> tokens around code segments
#     # Splitting by <comment> and adding <code> token to non-comment segments
#     segments = source_code.split("<comment>")
#     processed_code = ""
#     for i, segment in enumerate(segments):
#         if i % 2 == 0:  # Non-comment segment
#             processed_code += f"<code> {segment} </code>"
#         else:  # Comment segment
#             processed_code += f"<comment> {segment} </comment>"

#     return processed_code

def convert_examples_to_features(func, label, cpg_data, tokenizer, args):
    # if args.use_word_level_tokenizer:
    #     encoded = tokenizer.encode(func)
    #     encoded = encoded.ids
    #     if len(encoded) > 510:
    #         encoded = encoded[:510]
    #     encoded.insert(0, 0)
    #     encoded.append(2)
    #     if len(encoded) < 512:
    #         padding = 512 - len(encoded)
    #         for _ in range(padding):
    #             encoded.append(1)
    #     source_ids = encoded
    #     source_tokens = []
    #     return InputFeatures(source_tokens, source_ids, label)
    # source
    classifier = joblib.load('class.py')
    lines = func.strip().split('\n')
    line_to_tokens = {}
    line_number = 0
    
    for i, line in enumerate(lines):
        tokens = tokenizer.tokenizer(line)
        # print(tokens)
        line_to_tokens[i+1] = tokens
        line_number = line_number + 1
    
    token_list = [token for tokens in line_to_tokens.values() for token in tokens]

    embed_tokens = []
    for i, _ in enumerate(lines):
        index = i+1
        input_ids = line_to_tokens[index]['input_ids']
        token_positions = list(range(len(input_ids)))
        positional_encoding = PositionalEncoding2D(len(input_ids), max_lines=len(lines), max_line_length=max(len(line.split()) for line in lines))
        pos_encoded_tensor = positional_encoding(input_ids, line_number, token_positions)


        type_embedding = TypeEmbedding(len(input_ids), num_types=9)
        type_annotations = type_embedding.classify_c_code(func)
        type_indices = [type_embedding.type_to_index(annotations[1]) for annotations in type_annotations]

        type_tensors = torch.tensor([type_indices[i]], dtype=torch.long)

        embed_tokens.append(type_embedding(type_tensors, input_ids), pos_encoded_tensor)

    token_to_index = {token: i for i, token in enumerate(token_list)}
    processor = CPGProcessor(token_to_index, line_to_tokens)
    # Map type indices to tokens
    connection_matrix = processor.process(cpg_data, args)
    # print(source_ids.shape)
    # print(type_embed_tokens.shape)
    # source_tokens = source_ids + type_embed_tokens
    return InputFeatures(embed_tokens, token_list, label, connection_matrix)

# Function to match AST nodes with tokens and add syntax type
def match_ast_nodes_with_tokens_and_syntax(tokens, node_info, syntax_predictions):
    matches = []
    for i, token in enumerate(tokens):
        for node in node_info:
            node_line = node['properties'].get('line', {}).get('@value', -1)
            node_col = node['properties'].get('column', {}).get('@value', -1)
            if token['line'] == node_line and token['column'] == node_col:
                matches.append({
                    'token': token,
                    'node': node,
                    'syntax_type': syntax_predictions[token['line'] - 1]  # Assuming 1-based line indexing
                })
                break
    return matches


# Function to predict the syntax type for each line
def predict_syntax_type(lines, classifier):
    predictions = []
    for line in lines:
        prediction = classifier.predict([line])
        predictions.append(prediction[0])
    return predictions


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels, connection_matrix) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels, connection_matrix=connection_matrix)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
                
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids, labels)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:,1]>best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    PrecisionRecallDisplay.from_predictions(y_trues, logits[:, 1], name='LineVul')
    plt.savefig(f'eval_precision_recall_{args.model_name}.pdf')

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in test_dataloader:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":best_threshold,
    }
    PrecisionRecallDisplay.from_predictions(y_trues, logits[:, 1], name="LineVul")
    plt.savefig(f'test_precision_recall_{args.model_name}.pdf')

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    logits = [l[1] for l in logits]
    # result_df = generate_result_df(logits, y_trues, y_preds, args)
    # sum_lines, sum_flaw_lines = get_line_statistics(result_df)
    
    # write raw predictions if needed
    if args.write_raw_preds:
        write_raw_preds_csv(args, y_preds)

    # define reasoning method
    if args.reasoning_method == "all":
            all_reasoning_method = ["attention", "lig", "saliency", "deeplift", "deeplift_shap", "gradient_shap"]
    else:
        all_reasoning_method = [args.reasoning_method]


def generate_result_df(logits, y_trues, y_preds, args):
    df = pd.read_json(args.test_data_file)
    all_num_lines = []
    all_processed_func = df["func"].tolist()
    for func in all_processed_func:
        all_num_lines.append(get_num_lines(func))
    flaw_line_indices = df["flaw_line_index"].tolist()
    all_num_flaw_lines = []
    total_flaw_lines = 0
    for indices in flaw_line_indices:
        if isinstance(indices, str):
            indices = indices.split(",")
            num_flaw_lines = len(indices)
            total_flaw_lines += num_flaw_lines
        else:
            num_flaw_lines = 0
        all_num_flaw_lines.append(num_flaw_lines)
    assert len(logits) == len(y_trues) == len(y_preds) == len(all_num_flaw_lines)
    return pd.DataFrame({"logits": logits, "y_trues": y_trues, "y_preds": y_preds, 
                         "index": list(range(len(logits))), "num_flaw_lines": all_num_flaw_lines, "num_lines": all_num_lines, 
                         "flaw_line": df["flaw_line"], "func": df["func"]})

def write_raw_preds_csv(args, y_preds):
    df = pd.read_json(args.test_data_file)
    df["raw_preds"] = y_preds
    df.to_csv("./results/raw_preds.csv", index=False)

def get_num_lines(func):
    func = func.split("\n")
    func = [line for line in func if len(line) > 0]
    return len(func)

def get_line_statistics(result_df):
    total_lines = sum(result_df["num_lines"].tolist())
    total_flaw_lines = sum(result_df["num_flaw_lines"].tolist())
    return total_lines, total_flaw_lines

def rank_lines(all_lines_score_with_label, is_attention, ascending_ranking):
    # flatten the list
    all_lines_score_with_label = [line for lines in all_lines_score_with_label for line in lines]
    if is_attention:
        all_scores = [line[0].item() for line in all_lines_score_with_label]
    else:
        all_scores = [line[0] for line in all_lines_score_with_label]
    all_labels = [line[1] for line in all_lines_score_with_label]
    rank_df = pd.DataFrame({"score": all_scores, "label": all_labels})
    rank_df = rank_dataframe(rank_df, "score", ascending_ranking)
    return len(rank_df), rank_df

def rank_dataframe(df, rank_by: str, ascending: bool):
    df = df.sort_values(by=[rank_by], ascending=ascending)
    df = df.reset_index(drop=True)
    return df

def top_k_effort(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="label"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += 1
        if rank_df[label_col_name][i] == 1:
            caught_flaw_line += 1
        if target_flaw_line == caught_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line

def top_k_effort_pred_prob(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="y_preds"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
        if caught_flaw_line >= target_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line

def top_k_recall(pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, top_k_loc):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    inspect_neg_lines = True
    for i in range(len(pos_rank_df)):
        inspected_line += 1
        if inspected_line > target_inspected_line:
            inspect_neg_lines = False
            break
        if pos_rank_df["label"][i] == 1 or pos_rank_df["label"][i] is True:
            caught_flaw_line += 1
    if inspect_neg_lines:
        for i in range(len(neg_rank_df)):
            inspected_line += 1
            if inspected_line > target_inspected_line:
                break
            if neg_rank_df["label"][i] == 1 or neg_rank_df["label"][i] is True:
                caught_flaw_line += 1
    return round(caught_flaw_line / sum_flaw_lines, 4)

def top_k_recall_pred_prob(rank_df, sum_lines: int, sum_flaw_lines: int, top_k_loc: float, label_col_name="y_preds"):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if inspected_line > target_inspected_line:
            break
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
    return round(caught_flaw_line / sum_flaw_lines, 4)

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id):
    seq_length = input_ids.size(1)
    ref_input_ids = [cls_token_id] + [ref_token_id] * (seq_length-2) + [sep_token_id]
    return torch.tensor([ref_input_ids])


def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--cpg_directory", default=None, type=str, help="This is the directory containning the CPGs json files.")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ") 
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int, # 8, 12, 16, 20  
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int, # 8, 12, 16, 20
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,# 0.001  1e-4   1e-5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1, # 10, 20, 
                        help="training epochs")
    # RQ2
    parser.add_argument("--effort_at_top_k", default=0.2, type=float,
                        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines")
    parser.add_argument("--top_k_recall_by_lines", default=0.01, type=float,
                        help="Recall@TopK percent, sorted by line scores")
    parser.add_argument("--top_k_recall_by_pred_prob", default=0.2, type=float,
                        help="Recall@TopK percent, sorted by prediction probabilities")

    parser.add_argument("--do_sorting_by_line_scores", default=False, action='store_true',
                        help="Whether to do sorting by line scores.")
    parser.add_argument("--do_sorting_by_pred_prob", default=False, action='store_true',
                        help="Whether to do sorting by prediction probabilities.")
    # RQ3 - line-level evaluation
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12, #8, 12, 16
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                            help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('./word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name) 
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)        
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
    model = BertModel(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()
