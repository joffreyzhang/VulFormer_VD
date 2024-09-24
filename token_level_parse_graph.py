import json
from transformers import RobertaTokenizer



# Example:
# source_code = """
# static int alloc_addbyter ( int output , FILE * data ) {\n struct asprintf * infop = ( struct asprintf * ) data ;\n unsigned char outc = ( unsigned char ) output ;\n if ( ! infop -> buffer ) {\n infop -> buffer = malloc ( 32 ) ;\n if ( ! infop -> buffer ) {\n infop -> fail = 1 ;\n return - 1 ;\n }\n infop -> alloc = 32 ;\n infop -> len = 0 ;\n }\n else if ( infop -> len + 1 >= infop -> alloc ) {\n char * newptr ;\n newptr = realloc ( infop -> buffer , infop -> alloc * 2 ) ;\n if ( ! newptr ) {\n infop -> fail = 1 ;\n return - 1 ;\n }\n infop -> buffer = newptr ;\n infop -> alloc *= 2 ;\n }\n infop -> buffer [ infop -> len ] = outc ;\n infop -> len ++ ;\n return outc ;
# """
class AdjacencyGenerate(source_code):
    
    tokenizer = RobertaTokenizer(vocab_file='./tokenizer_1-vocab.json',
                                merges_file='./tokenizer_1-merges.txt')

    encoded_input = tokenizer(source_code)

    # The encoded result
    print(encoded_input)


    lines = source_code.strip().split('\n')

    # Tokenizing each line and creating line-to-tokens dictionary
    line_to_tokens = {}
    for i, line in enumerate(lines):
        tokens = tokenizer(line)
        line_to_tokens[i + 1] = tokens

    # Print the original lines and their tokenized form
    print("This is the line level:")
    for line_number, tokens in line_to_tokens.items():
        print(f"Line {line_number}: {tokens}")



    def extract_control_flow_edges(vertices, edges):
        vertex_to_line = {}
        control_flow_edges = []

        # Map vertex IDs to line numbers
        for vertex in vertices:
            if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number

        # Find control flow relationships from edges
        for edge in edges:
            if edge["label"] == "CFG":
                in_vertex = edge["inV"]["@value"]
                out_vertex = edge["outV"]["@value"]
                if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
                    edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
                if edge_tuple not in control_flow_edges:

                    control_flow_edges.append(edge_tuple)

        return control_flow_edges




    def extract_control_dependency_edges(vertices, edges):
        vertex_to_line = {}
        control_flow_edges = []

        # Map vertex IDs to line numbers
        for vertex in vertices:
            if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number

        # Find control flow relationships from edges
        for edge in edges:
            if edge["label"] == "CDG":
                in_vertex = edge["inV"]["@value"]
                out_vertex = edge["outV"]["@value"]
                if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
                    edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
                if edge_tuple not in control_flow_edges:

                    control_flow_edges.append(edge_tuple)

        return control_flow_edges

    def extract_data_dependency_edges(vertices, edges):
        vertex_to_line = {}
        control_flow_edges = []

        # Map vertex IDs to line numbers
        for vertex in vertices:
            if vertex["label"] == "CALL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "IDENTIFIER" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number
            if vertex["label"] == "LITERAL" and "LINE_NUMBER" in vertex["properties"]:
                vertex_id = vertex["id"]["@value"]
                line_number = vertex["properties"]["LINE_NUMBER"]["@value"]["@value"]
                vertex_to_line[vertex_id] = line_number

        # Find control flow relationships from edges
        for edge in edges:
            if edge["label"] == "REACHING_DEF":
                in_vertex = edge["inV"]["@value"]
                out_vertex = edge["outV"]["@value"]
                if in_vertex in vertex_to_line and out_vertex in vertex_to_line:
                    edge_tuple = (vertex_to_line[out_vertex], vertex_to_line[in_vertex])
                if edge_tuple not in control_flow_edges:

                    control_flow_edges.append(edge_tuple)

        return control_flow_edges

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

    # Path to the JSON file
    json_file = '/content/drive/MyDrive/Colab Notebooks/drive-download-20231211T034526Z-001/example_dfg.json'

    # Run the program
    control_flow_edges, control_dependency_edges, data_dependency_edges = conbine_matrixs(json_file)
    print(control_flow_edges)


    line_to_token = {}
    def remove_duplicates_preserve_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    line_to_token = {line: data['input_ids'] for line, data in line_to_tokens.items()}

    print(line_to_token)

    # for line, data in line_to_tokens.items():
    #     tokens = tokenizer.convert_ids_to_tokens(data['input_ids'])
    #     line_to_token[line] = tokens


    token_list = [token for tokens in line_to_token.values() for token in tokens]
    print(token_list)

    # Create a mapping from tokens to indices
    token_to_index = {token: i for i, token in enumerate(token_list)}

    # Initialize the adjacency matrix
    num_tokens = len(token_list)

    def create_adjacency_matrix(num_tokens, edges, weight):
        """Create an adjacency matrix for a given type of edges with a specified weight."""
        matrix = [[0] * num_tokens for _ in range(num_tokens)]
        for from_line, to_line in edges:
            for from_token in line_to_token[from_line]:
                for to_token in line_to_token[to_line]:
                    i = token_to_index[from_token]
                    j = token_to_index[to_token]
                    matrix[i][j] = weight
        return matrix
    control_flow_weight = 2  # Assuming control flow edges have a weight of 2
    control_dependency_weight = 3  # Control dependency weight of 3
    data_dependency_weight = 7  # Data dependency weight of 7

    # Create separate adjacency matrices for each type of edge
    control_flow_matrix = create_adjacency_matrix(num_tokens, control_flow_edges, control_flow_weight)
    control_dependency_matrix = create_adjacency_matrix(num_tokens, control_dependency_edges, control_dependency_weight)
    data_dependency_matrix = create_adjacency_matrix(num_tokens, data_dependency_edges, data_dependency_weight)

    # Sum the matrices
    combined_matrix = [[control_flow_matrix[i][j] + control_dependency_matrix[i][j] + data_dependency_matrix[i][j] 
                        for j in range(num_tokens)] for i in range(num_tokens)]

    # Example output
    for i in range(len(combined_matrix)):
        print(combined_matrix[i])
    # Populate the adjacency matrix based on control flow edges


    