import numpy as np
import pandas as pd
import datetime
import phonenumbers

label_map = {
    'O' : 0,
    'B-NAME_STUDENT': 1,
    'I-NAME_STUDENT': 2, 
    'B-URL_PERSONAL': 3, 
    'I-URL_PERSONAL': 4, 
    'B-ID_NUM': 5, 
    'I-ID_NUM': 6,
    'B-EMAIL': 7, 
    'I-EMAIL': 8,
    'B-STREET_ADDRESS': 9, 
    'I-STREET_ADDRESS': 10,
    'B-PHONE_NUM': 11, 
    'I-PHONE_NUM': 12, 
    'B-USERNAME': 13,
    'I-USERNAME': 14,
}
reverse_label_map = {v: k for k, v in label_map.items()}

def is_phone_number(possible_phone_num):
    try:
        z = phonenumbers.parse(possible_phone_num, 'US')
        if phonenumbers.is_possible_number(z):
            return True
    except:
        return False
    
data_df = pd.read_parquet('predicted_results_20240421_100936.parquet', engine='pyarrow')
all_predicted_labels = list(data_df['all_predicted_labels'])
all_true_labels = list(data_df['all_true_labels'])
all_attention_masks = list(data_df['all_attention_masks'])
aggregated_predicted_labels = []
aggregated_true_labels = []
aggregated_attention_masks = []
aggregated_tokens = []
test_df = pd.read_parquet('test_data.parquet', engine='pyarrow')
for i, (index_mapping, token_mapping) in enumerate(zip(test_df['index_mappings'], test_df['token_mappings'])):
    # length = max(index_mapping) + 1
    length = 512
    predicted_labels = [0] * length
    true_labels = [0] * length
    attention_masks = [0] * length
    restored_tokens = [""] * length
    
    for j, (index, token) in enumerate(zip(index_mapping, token_mapping)):
        if index < 0:
            continue
        if true_labels[index] == 0:
            true_labels[index] = all_true_labels[i][j]
            predicted_labels[index] = all_predicted_labels[i][j]
            attention_masks[index] = all_attention_masks[i][j]
            restored_tokens[index] = token
    # post process 1
    # first label should start with B-
    for k in range(len(predicted_labels)):
        if k == 0:
            continue
        if reverse_label_map[predicted_labels[k]].startswith('I-'):
            base_label = reverse_label_map[predicted_labels[k]][2:]
            if reverse_label_map[predicted_labels[k-1]].startswith('O') or reverse_label_map[predicted_labels[k-1]][2:] != base_label:
                predicted_labels[k] -= 1
    # post process 2
    # phone_num <-> street_address
    k = 0
    while k < len(predicted_labels):
        # Check if the current element is 9 and the next element is 10
        if predicted_labels[k] == 9:
            start = k
            # Move forward to find all subsequent 10s
            k += 1
            while k < len(predicted_labels) and predicted_labels[k] == 10:
                k += 1
            number = ""
            for j in range(start, k):
                number += restored_tokens[j]
            if is_phone_number(number):
                predicted_labels[start] = 11
                for j in range(start + 1, k):
                    predicted_labels[j] = 12
        else:
            k += 1
    aggregated_predicted_labels.append(predicted_labels)
    aggregated_true_labels.append(true_labels)
    aggregated_attention_masks.append(attention_masks)

# Flatten the lists
aggregated_true_labels = [label for sublist in aggregated_true_labels for label in sublist]
aggregated_attention_masks = [mask for sublist in aggregated_attention_masks for mask in sublist]
aggregated_predicted_labels = [label for sublist in aggregated_predicted_labels for label in sublist]

data = {
    "aggregated_true_labels": aggregated_true_labels,
    "aggregated_attention_masks": aggregated_attention_masks,
    "aggregated_predicted_labels": aggregated_predicted_labels
}

data_df = pd.DataFrame(data)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
data_df.to_parquet(f'postprocess_results_{timestamp}.parquet', engine='pyarrow')
