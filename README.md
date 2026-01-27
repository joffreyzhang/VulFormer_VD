# CosFormer: Vulnerability Detection with Code Property Graphs

A deep learning system for detecting vulnerabilities in C/C++ source code using transformer models enhanced with Code Property Graph (CPG) information.

## Overview

CosFormer combines pre-trained transformer models (RoBERTa/CodeBERT) with structural code information from Code Property Graphs to classify functions as vulnerable or non-vulnerable. The system also provides line-level vulnerability localization to identify specific vulnerable code lines.

## Key Features

- **Multi-modal Architecture**: Combines token embeddings, positional encoding, syntax type encoding, and CPG edge encoding
- **CPG Integration**: Leverages Control Flow Graph (CFG), Control Dependency Graph (CDG), and Data Dependency Graph (DDG) relationships
- **Line-level Localization**: Identifies specific vulnerable lines within functions
- **Interpretability**: Supports multiple explanation methods (attention, integrated gradients, saliency, DeepLift, SHAP)
- **Pre-trained Models**: Built on RoBERTa/CodeBERT for strong code understanding

## Architecture

The system uses a multi-modal approach combining:

1. **Token Embeddings**: Pre-trained RoBERTa/CodeBERT embeddings for code tokens
2. **2D Positional Encoding**: Captures both line number and token position within lines
3. **Type Encoding**: Classifies each line into 9 C/C++ syntax categories:
   - Selection statements (if, switch)
   - Loop statements (for, while, do-while)
   - Jump statements (break, continue, return, goto)
   - Assignment statements
   - Defining statements (variable/function declarations)
   - Function calls
   - Comments
   - Blank lines
   - Other
4. **CPG Edge Encoding**: Incorporates graph structure from CFG, CDG, and DDG
5. **Degree Encoding**: Encodes in-degree and out-degree information from the CPG

### Key Components

- `main.py` - Entry point with training/evaluation loops, dataset loading, and CLI argument parsing
- `model.py` - Custom BERT model with CPG-aware self-attention mechanisms
- `edge_encoding.py` - CPGProcessor class that builds token-level adjacency matrices from CPG JSON files
- `positional_encoding.py` - PositionalEncoding2D for line/token position encoding
- `degree_encoding.py` - DegreeEmbedding for encoding node degree information
- `type_encoding.py` - TypeEmbedding for C/C++ syntax classification
- `data_process.py` - Data preprocessing utilities
- `token_level_parse_graph.py` - Token-level CPG parsing utilities

## Installation

### Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- tokenizers
- pandas
- numpy
- scikit-learn
- captum
- matplotlib
- joblib
- tqdm

### Install Dependencies

```bash
pip install torch transformers tokenizers pandas numpy scikit-learn captum matplotlib joblib tqdm
```

## Dataset Format

### Input JSON Format

The dataset should be in JSON format with the following fields:

```json
{
  "func": "int foo(int x) {\n  if (x < 0) return -1;\n  return x * 2;\n}",
  "target": 0,
  "flaw_line_index": ""
}
```

- `func`: C/C++ function source code
- `target`: Binary label (0 = non-vulnerable, 1 = vulnerable)
- `flaw_line_index`: Comma-separated line numbers of vulnerabilities (for vulnerable functions)

### CPG Files

Code Property Graph files must be pre-generated (e.g., using [Joern](https://joern.io/)) and placed at:
```
{cpg_directory}/{index}.c/export.json
```

Where `{index}` corresponds to the row index in the dataset JSON file.

## Usage

### Training

```bash
python main.py \
  --do_train \
  --train_data_file path/to/train.json \
  --eval_data_file path/to/eval.json \
  --cpg_directory path/to/cpg_files \
  --output_dir ./saved_models \
  --model_name_or_path microsoft/codebert-base \
  --tokenizer_name microsoft/codebert-base \
  --epochs 10 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --seed 123456
```

### Evaluation

```bash
python main.py \
  --do_test \
  --test_data_file path/to/test.json \
  --cpg_directory path/to/cpg_files \
  --output_dir ./saved_models \
  --model_name model.bin \
  --eval_batch_size 16
```

### Line-level Vulnerability Localization

```bash
python main.py \
  --do_test \
  --do_local_explanation \
  --test_data_file path/to/test.json \
  --cpg_directory path/to/cpg_files \
  --output_dir ./saved_models \
  --model_name model.bin \
  --reasoning_method attention
```

## Command-line Arguments

### Data Arguments
- `--train_data_file`: Path to training data JSON file
- `--eval_data_file`: Path to evaluation data JSON file
- `--test_data_file`: Path to test data JSON file
- `--cpg_directory`: Directory containing CPG files

### Model Arguments
- `--model_name_or_path`: Pre-trained model name or path (default: `microsoft/codebert-base`)
- `--tokenizer_name`: Tokenizer name or path
- `--model_name`: Saved model filename for testing
- `--num_attention_heads`: Number of attention heads (default: 12)

### Training Arguments
- `--epochs`: Number of training epochs (default: 10)
- `--train_batch_size`: Training batch size (default: 16)
- `--eval_batch_size`: Evaluation batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `--seed`: Random seed (default: 123456)

### Interpretability Arguments
- `--reasoning_method`: Explanation method (`attention`, `lig`, `saliency`, `deeplift`, `deeplift_shap`, `gradient_shap`)
- `--do_local_explanation`: Enable line-level vulnerability localization
- `--write_raw_preds`: Output raw predictions to CSV file

### Execution Modes
- `--do_train`: Run training
- `--do_test`: Run testing/evaluation

## Output

The system produces:

1. **Model Checkpoints**: Saved in `{output_dir}/checkpoint-best-f1/model.bin`
2. **Predictions**: CSV file with predictions (if `--write_raw_preds` is enabled)
3. **Metrics**: Accuracy, Precision, Recall, F1-score
4. **Line-level Scores**: Vulnerability scores for each line (if `--do_local_explanation` is enabled)

## Data Flow

1. Load JSON dataset with function source code, labels, and flaw line indices
2. Filter functions to those with fewer than 100 lines
3. Load corresponding CPG from `{cpg_directory}/{index}.c/export.json`
4. Tokenize code line-by-line using RoBERTa/CodeBERT tokenizer
5. Generate positional encodings (2D: line number + token position)
6. Generate type encodings (C/C++ syntax classification)
7. Generate edge encodings (CPG adjacency matrix at token level)
8. Generate degree encodings (in-degree and out-degree)
9. Combine all encodings and pass through transformer model
10. Output binary classification + line-level vulnerability scores

## Model Architecture Details

The model extends RoBERTa with:

- **Custom Self-Attention**: Modified to incorporate CPG edge information
- **Multi-head Attention**: Configurable number of attention heads
- **Classification Head**: Binary classification (vulnerable/non-vulnerable)
- **Token Classification Head**: Line-level vulnerability scoring
- **Embedding Layers**: Positional, type, and degree embeddings added to token embeddings

## Datasets

The system has been tested with:

- **Big-Vul**: Large-scale vulnerability dataset
- **Devign**: Function-level vulnerability detection dataset
- **Reveal**: Real-world vulnerability dataset

Dataset directories are included in the repository structure.

## Known Limitations

- Functions must be fewer than 100 lines
- Requires pre-generated CPG files (not included in training pipeline)
- Some dependencies (`linevul_model.py`, `class.py`) may need to be obtained separately

## Citation

If you use this code in your research, please cite the relevant papers on CosFormer and vulnerability detection.

## License

Please refer to the repository license file for usage terms.

## Contributing

Contributions are welcome. Please ensure code follows the existing structure and includes appropriate documentation.

## Contact

For questions or issues, please open an issue in the repository.
