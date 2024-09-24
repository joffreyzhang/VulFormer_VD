import json
import re
# class TransformerWithEdgeEncoding(nn.Module):
#     def __init__(self, num_tokens, num_edge_types, embed_size, edge_embed_size):
#         super().__init__()
#         self.token_embeddings = nn.Embedding(num_tokens, embed_size)
#         self.edge_embeddings = nn.Embedding(num_edge_types, edge_embed_size)
       

#     def forward(self, input_ids, edge_ids):
        
#         token_embeds = self.token_embeddings(input_ids)
#         edge_embeds = self.edge_embeddings(edge_ids)

        
# #         return output

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
# tokenizer = RobertaTokenizer(vocab_file='E:/SE/text/VulFormer/VulFormer/bpe_tokenizer/bpe_tokenizer-vocab.json',
#                                 merges_file='E:/SE/text/VulFormer/VulFormer/bpe_tokenizer/bpe_tokenizer-merges.txt')
# with open('E:/SE/VulFormer/data/Devign/101.c','r') as file:
#     source_code = file.read()
# processed_code = preprocess_source_code(source_code)
# encoded_input = tokenizer(processed_code)


# lines = source_code.strip().split('\n')


# # Tokenizing each line and creating line-to-tokens dictionary
# line_to_tokens = {}
# for i, line in enumerate(lines):
#     tokens = tokenizer(line)
#     line_to_tokens[i + 1] = tokens

# # Print the original lines and their tokenized form
# print("This is the line level:")
# for line_number, tokens in line_to_tokens.items():
#     print(f"Line {line_number}: {tokens}")

# def extract_control_flow_edges(vertices, edges):
#     vertex_to_line = {}
#     control_flow_edges = []

#         # Map vertex IDs to line numbers
#     for vertex in vertices:
#         if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number

#         # Find control flow relationships from edges
#     for edge in edges:
#         if edge["label"] == "CFG":
#             in_vertex = edge["inV"]["@value"]
#             out_vertex = edge["outV"]["@value"]
#             edge_tuple = None
#             if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
#                 edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
#             if edge_tuple not in control_flow_edges:

#                 control_flow_edges.append(edge_tuple)

#     return control_flow_edges


# def extract_control_dependency_edges(vertices, edges):
#     vertex_to_line = {}
#     control_dependency_edges = []

#     # Map vertex IDs to line numbers
#     for vertex in vertices:
#         if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number

#         # Find control flow relationships from edges
#     for edge in edges:
#         if edge["label"] == "CDG":
#             in_vertex = edge["inV"]["@value"]
#             out_vertex = edge["outV"]["@value"]
#             edge_tuple = None
#             if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
#                 edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
#             if edge_tuple not in control_dependency_edges:

#                 control_dependency_edges.append(edge_tuple)

#     return control_dependency_edges


# def extract_data_dependency_edges(vertices, edges):
#     vertex_to_line = {}
#     data_dependency_edges = []

#     # Map vertex IDs to line numbers
#     for vertex in vertices:
#         if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number
#         if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
#             vertex_id = vertex["id"]["@value"]
#             line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
#             vertex_to_line[vertex_id] = line_number

#         # Find control flow relationships from edges
#     for edge in edges:
#         if edge["label"] == "REACHING_DEF":
#             in_vertex = edge["inV"]["@value"]
#             out_vertex = edge["outV"]["@value"]
#             edge_tuple = None
#             if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
#                 edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
#             if edge_tuple not in control_flow_edges:

#                 data_dependency_edges.append(edge_tuple)

#     return data_dependency_edges

# def conbine_matrixs(json_file_path):
#     # Read the JSON file
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)

#     # Extract vertices and edges
#     graph_elements = data.get('@value', {})
#     edges = graph_elements.get('edges', [])
#     vertices = graph_elements.get('vertices', [])

#     # Extract control flow edges
#     control_flow_edges = extract_control_flow_edges(vertices, edges)
#     control_dependency_edges = extract_control_dependency_edges(vertices,edges)
#     data_dependency_edges = extract_data_dependency_edges(vertices, edges)
#     return control_flow_edges, control_dependency_edges, data_dependency_edges

# json_file = './data/Devign/export.json'


# line_to_token = {}
# def remove_duplicates_preserve_order(seq):
#     seen = set()
#     seen_add = seen.add
#     return [x for x in seq if not (x in seen or seen_add(x))]

# line_to_token = {line: data['input_ids'] for line, data in line_to_tokens.items()}

# print(line_to_token)

# # for line, data in line_to_tokens.items():
# #     tokens = tokenizer.convert_ids_to_tokens(data['input_ids'])
# #     line_to_token[line] = tokens


# token_list = [token for tokens in line_to_token.values() for token in tokens]
# print(token_list)

# # Create a mapping from tokens to indices
# token_to_index = {token: i for i, token in enumerate(token_list)}

#     # Initialize the adjacency matrix
# num_tokens = len(token_list)

# def create_adjacency_matrix(num_tokens, edges, weight):
#     """Create an adjacency matrix for a given type of edges with a specified weight."""
#     matrix = [[0] * num_tokens for _ in range(num_tokens)]

#     for edge in edges:
#         # Check if edge is not None and is iterable with exactly two elements
#         if edge is not None and isinstance(edge, (list, tuple)) and len(edge) == 2:
#             from_line, to_line = edge

#             for from_token in line_to_token.get(from_line, []):
#                 for to_token in line_to_token.get(to_line, []):
#                     i = token_to_index.get(from_token)
#                     j = token_to_index.get(to_token)

#                     # Check if both indices are valid before updating the matrix
#                     if i is not None and j is not None:
#                         matrix[i][j] = weight

#     return matrix

# control_flow_weight = 2  # Assuming control flow edges have a weight of 2
# control_dependency_weight = 3  # Control dependency weight of 3
# data_dependency_weight = 7  # Data dependency weight of 7
# control_flow_edges = []
# control_dependency_edges = []
# data_dependency_edges = []
# control_flow_edges, control_dependency_edges, data_dependency_edges = conbine_matrixs(json_file)

# # Create separate adjacency matrices for each type of edge
# control_flow_matrix = create_adjacency_matrix(num_tokens, control_flow_edges, control_flow_weight)
# control_dependency_matrix = create_adjacency_matrix(num_tokens, control_dependency_edges, control_dependency_weight)
# data_dependency_matrix = create_adjacency_matrix(num_tokens, data_dependency_edges, data_dependency_weight)

# # Sum the matrices
# combined_matrix = [[control_flow_matrix[i][j] + control_dependency_matrix[i][j] + data_dependency_matrix[i][j] 
#                     for j in range(num_tokens)] for i in range(num_tokens)]
# print(len(combined_matrix))
# # Example output
# for i in range(len(combined_matrix)):
#     print(combined_matrix[i])
import json

class CPGProcessor:
    def __init__(self, token_to_index, line_to_token):
        self.token_to_index = token_to_index
        self.line_to_token = line_to_token

    def remove_duplicates_preserve_order(self, seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    def extract_edges(self, vertices, edges, edge_label):
        vertex_to_line = {}
        extracted_edges = []

        # Map vertex IDs to line numbers
        for vertex in vertices:
            if vertex["label"] in ["CALL", "IDENTIFIER", "LITERAL"] and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number

        # Find edges from the specified label
        for edge in edges:
            if edge["label"] == edge_label:
                in_vertex = edge["inV"]["@value"]
                out_vertex = edge["outV"]["@value"]
                if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
                    edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
                    if edge_tuple not in extracted_edges:
                        extracted_edges.append(edge_tuple)

        return extracted_edges

    def create_adjacency_matrix(self, num_tokens, edges, weight):
        matrix = [[0] * num_tokens for _ in range(num_tokens)]

        for edge in edges:
            from_line, to_line = edge
            for from_token in self.line_to_token.get(from_line, []):
                for to_token in self.line_to_token.get(to_line, []):
                    i = self.token_to_index.get(from_token)
                    j = self.token_to_index.get(to_token)
                    if i is not None and j is not None:
                        matrix[i][j] = weight

        return matrix

    def process(self, cpg_data, control_flow_weight=2, control_dependency_weight=3, data_dependency_weight=7):
        data = cpg_data

        graph_elements = data.get('@value', {})
        edges = graph_elements.get('edges', [])
        vertices = graph_elements.get('vertices', [])

        control_flow_edges = self.extract_edges(vertices, edges, "CFG")
        control_dependency_edges = self.extract_edges(vertices, edges, "CDG")
        data_dependency_edges = self.extract_edges(vertices, edges, "REACHING_DEF")

        token_list = [token for tokens in self.line_to_token.values() for token in tokens]
        token_list = self.remove_duplicates_preserve_order(token_list)
        num_tokens = len(token_list)

        control_flow_matrix = self.create_adjacency_matrix(num_tokens, control_flow_edges, control_flow_weight)
        control_dependency_matrix = self.create_adjacency_matrix(num_tokens, control_dependency_edges, control_dependency_weight)
        data_dependency_matrix = self.create_adjacency_matrix(num_tokens, data_dependency_edges, data_dependency_weight)

        # Sum the matrices
        combined_matrix = [[control_flow_matrix[i][j] + control_dependency_matrix[i][j] + data_dependency_matrix[i][j] 
                            for j in range(num_tokens)] for i in range(num_tokens)]
        return combined_matrix

