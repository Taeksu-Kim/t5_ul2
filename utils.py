import pickle
import numpy as np

def read_pickle(path_list):
    examples = []
    # max_len = 512

    for encoding_file in path_list:
        with open(encoding_file, "rb") as f:
            pickle_data = pickle.load(f)
            
        examples.extend(pickle_data)
    
    return np.array(examples, dtype=object)

    # fixed_examples = []

    # for i in range(len(examples)):
    #     if len(examples[i]['input_ids']) > 128:
    #         fixed_examples.append(examples[i]['input_ids'][:-1][:max_len])

    # del examples

    # fixed_examples = np.array(fixed_examples, dtype=object)

    # return fixed_examples