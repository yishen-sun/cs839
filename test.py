
import numpy as np
import pandas as pd
import datetime

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


data_df = pd.read_parquet('postprocess_results_20240501_095547.parquet', engine='pyarrow')
aggregated_true_labels = list(data_df['aggregated_true_labels'])
aggregated_attention_masks = list(data_df['aggregated_attention_masks'])
aggregated_predicted_labels = list(data_df['aggregated_predicted_labels'])


from sklearn.metrics import precision_score, recall_score, f1_score
def calculate_metrics(true_labels, predicted_labels, labels):
    precision = precision_score(true_labels, predicted_labels, average=None, labels=labels)
    recall = recall_score(true_labels, predicted_labels, average=None, labels=labels)
    f1 = f1_score(true_labels, predicted_labels, average=None, labels=labels)
    # Custom F5 calculation
    f5 = (1 + 5**2) * (precision * recall) / ((5**2 * precision) + recall + 1)
    number = np.asarray(true_labels).shape
    
    return precision, recall, f1, f5

# Calculate weighted averages
def calculate_metrics_weighted(true_labels, predicted_labels, labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted', labels=labels)
    recall = recall_score(true_labels, predicted_labels, average='weighted', labels=labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', labels=labels)
    # Custom F5 calculation
    f5 = (1 + 5**2) * (precision * recall) / ((5**2 * precision) + recall + 1)

    return precision, recall, f1, f5

def calculate_metrics_micro(true_labels, predicted_labels, labels):
    precision = precision_score(true_labels, predicted_labels, average='micro', labels=labels)
    recall = recall_score(true_labels, predicted_labels, average='micro', labels=labels)
    f1 = f1_score(true_labels, predicted_labels, average='micro', labels=labels)
    # Custom F5 calculation
    f5 = (1 + 5**2) * (precision * recall) / ((5**2 * precision) + recall + 1)

    return precision, recall, f1, f5

# You'll need to preprocess your predictions and true_labels to filter out padding based on attention masks and flatten them for sklearn metrics
# true_labels, predicted_labels should be flattened lists of labels for tokens where the corresponding attention_mask is 1
labels = list(range(1, 15))
precision, recall, f1, f5 = calculate_metrics(aggregated_true_labels, aggregated_predicted_labels, labels)

w_pre, w_recall, w_f1, w_f5 = calculate_metrics_weighted(aggregated_true_labels, aggregated_predicted_labels, labels)
micro_pre, micro_recall, micro_f1, micro_f5 = calculate_metrics_micro(aggregated_true_labels, aggregated_predicted_labels, labels)

# Get the current date and time as a string in the format "YYYYMMDD_HHMMSS"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Append the timestamp to the filename
filename = f"result_file_{timestamp}.txt"

# Open a file with the timestamped name to write the results
with open(filename, 'w') as file:
    # Write the scores for each label to the file
    for i, label_id in enumerate(labels):
        print(f"Label {reverse_label_map[label_id]} - Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1[i]}, F5: {f5[i]}")
        file.write(f"Label {reverse_label_map[label_id]} - Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1[i]}, F5: {f5[i]}\n")
    
    # Calculate unweighted (macro) averages
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    average_f1 = np.mean(f1)
    average_f5 = np.mean(f5)
    
    print(f"weighted Precision: {w_pre}")
    print(f"weighted Recall: {w_recall}")
    print(f"weighted F1: {w_f1}")
    print(f"weighted F5: {w_f5}")
    
    file.write(f"{16_5e-5}\n")
    file.write(f"Average weighted Precision: {w_pre}\n")
    file.write(f"Average weighted Recall: {w_recall}\n")
    file.write(f"Average weighted F1: {w_f1}\n")
    file.write(f"Average weighted F5: {w_f5}\n")
    
    file.write(f"Micro Precision: {micro_pre}\n")
    file.write(f"Micro Recall: {micro_recall}\n")
    file.write(f"Micro F1: {micro_f1}\n")
    file.write(f"Micro F5: {micro_f5}\n")
