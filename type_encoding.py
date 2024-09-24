import torch.nn as nn 
import torch


class TypeEmbedding(nn.Module):
    def __init__(self, embed_size, num_types):
        super(TypeEmbedding, self).__init__()
        self.type_embeddings = nn.Embedding(num_types, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, token_embeddings, c_code):
        token_embeddings = token_embeddings.unsqueeze(0) 
        seq_len = token_embeddings.size(1)
        type_list = self.classify_c_code(c_code)
        type_indices = torch.tensor([self.type_to_index(t) for t in type_list[:seq_len]])

        type_embeds = self.type_embeddings(type_indices)
        print(token_embeddings)
        print(type_embeds)
        type_encoded = token_embeddings + type_embeds

        return self.norm(type_encoded)

    def classify_c_code(self, c_code):
        lines = c_code.split('\n')
        classified_lines = []
        
        open_braces_count = 0
        in_multiline_statement = False
        prev_category = "other"
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            prev_line = lines[i - 1].strip() if i > 0 else ""
            open_braces_count += stripped_line.count('{') - stripped_line.count('}')
            if open_braces_count == 0 and in_multiline_statement:
                in_multiline_statement = False
            # Classify based on the start of the line
            if stripped_line.startswith(("if ", "else", "switch ", "case ", "default ")):
                category = "selection_statement"
            elif stripped_line.startswith(("while ", "do ", "for ")):
                category = "loop_statement"
            elif stripped_line.startswith(("goto ", "continue", "break", "return")):
                category = "jump_statement"
            elif stripped_line.startswith(("//", "/*")) or "*/" in stripped_line:
                category = "comment"
            elif "=" in stripped_line and not stripped_line.startswith(("if ", "while ", "for ")) and not in_multiline_statement:
                category = "assignment_statement"
            elif stripped_line.endswith(";") and any(dtype in stripped_line for dtype in ["int ", "char ", "float ", "double ", "struct ", "unsigned"]):
                category = "defining_statement"
            elif stripped_line.endswith(");") and "(" in stripped_line and ";" in stripped_line:
                category = "function_call"
            elif stripped_line == "":
                category = "blank_line"
            elif stripped_line.endswith("{") and not in_multiline_statement:
                # Use additional checks here to refine classification
                if any(stripped_line.startswith(kw) for kw in ("if ", "else", "switch ", "for ", "while ", "do ")):
                    category = "selection_statement" if stripped_line.startswith(("if ", "else", "switch ")) else "loop_statement"
                    in_multiline_statement = True
                else:
                    category = "other"
            elif open_braces_count > 0 or in_multiline_statement:
                category = "continuation_of_multiline"
            else:
                category = "other"
            if stripped_line == "{" and not prev_line.endswith((")", ";")):
                category = "other"

            if category == "continuation_of_multiline":
                category = prev_category
            else:
                prev_category = category
            classified_lines.append((stripped_line, category))

        return classified_lines


    
    
    def type_to_index(self, type_name):
        # Map each type name to an index
        type_index_mapping = {
            "selection_statement": 0,
            "loop_statement": 1,
            "jump_statement": 2,
            "comment": 3,
            "assignment_statement": 4,
            "defining_statement": 5,
            "function_call": 6,
            "blank_line": 7,
            "other": 8,
            "continuation_of_multiline": 9
        }
        return type_index_mapping.get(type_name, type_index_mapping["other"])


def classify_c_code(c_code):
    lines = c_code.split('\n')
    classified_lines = []
    
    open_braces_count = 0
    in_multiline_statement = False
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        prev_line = lines[i - 1].strip() if i > 0 else ""
        open_braces_count += stripped_line.count('{') - stripped_line.count('}')
        if open_braces_count == 0 and in_multiline_statement:
            in_multiline_statement = False
        # Classify based on the start of the line
        if stripped_line.startswith(("if ", "else", "switch ", "case ", "default ")):
            category = "selection_statement"
        elif stripped_line.startswith(("while ", "do ", "for ")):
            category = "loop_statement"
        elif stripped_line.startswith(("goto ", "continue", "break", "return")):
            category = "jump_statement"
        elif stripped_line.startswith(("//", "/*")) or "*/" in stripped_line:
            category = "comment"
        elif "=" in stripped_line and not stripped_line.startswith(("if ", "while ", "for ")) and not in_multiline_statement:
            category = "assignment_statement"
        elif stripped_line.endswith(";") and any(dtype in stripped_line for dtype in ["int ", "char ", "float ", "double ", "struct ", "unsigned"]):
            category = "defining_statement"
        elif stripped_line.endswith(");") and "(" in stripped_line and ";" in stripped_line:
            category = "function_call"
        elif stripped_line == "":
            category = "blank_line"
        elif stripped_line.endswith("{") and not in_multiline_statement:
            # Use additional checks here to refine classification
            if any(stripped_line.startswith(kw) for kw in ("if ", "else", "switch ", "for ", "while ", "do ")):
                category = "selection_statement" if stripped_line.startswith(("if ", "else", "switch ")) else "loop_statement"
                in_multiline_statement = True
            else:
                category = "other"
        elif open_braces_count > 0 or in_multiline_statement:
            category = "continuation_of_multiline"
        else:
            category = "other"
        if stripped_line == "{" and not prev_line.endswith((")", ";")):
            category = "other"

        classified_lines.append((stripped_line, category))

    return classified_lines

    