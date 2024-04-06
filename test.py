from transformers import TFBertForTokenClassification
import numpy as np
import pandas as pd
import tensorflow as tf

def df_to_tfdata(df, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': list(df['input_ids']),
            'attention_mask': list(df['attention_mask'])
        },
        list(df['numeric_labels'])
    ))
    dataset = dataset.batch(batch_size)
    return dataset

model = TFBertForTokenClassification.from_pretrained("my_pii_detection_model", num_labels=15)

test_df = pd.read_parquet('test_data.parquet', engine='pyarrow')
test_df.info()
test_dataset = df_to_tfdata(test_df)
predictions = model.predict(test_dataset)  # Assuming test_dataset is already prepared
predicted_label_indices = np.argmax(predictions.logits, axis=-1)

true_labels = []
attention_masks = []
for batch in test_dataset.unbatch():  # Iterate over each batch in the dataset
    true_labels.append(batch[1].numpy())  # Assuming the second element of the batch is numeric_labels
    attention_masks.append(batch[0]['attention_mask'].numpy())  # Assuming attention_mask is part of the input dict
# Flatten the lists
true_labels = [label for sublist in true_labels for label in sublist]
attention_masks = [mask for sublist in attention_masks for mask in sublist]
predicted_labels_flat = [label for sublist in predicted_label_indices for label in sublist]
# Filter Based on Attention Masks
filtered_true_labels = []
filtered_predicted_labels = []
for i in range(len(attention_masks)):
    if attention_masks[i] == 1:  # If the token is not a padding token
        filtered_true_labels.append(true_labels[i])
        filtered_predicted_labels.append(predicted_labels_flat[i])

from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(true_labels, predicted_labels, labels):
    precision = precision_score(true_labels, predicted_labels, average=None, labels=labels)
    recall = recall_score(true_labels, predicted_labels, average=None, labels=labels)
    f1 = f1_score(true_labels, predicted_labels, average=None, labels=labels)
    # Custom F5 calculation
    f5 = (1 + 5**2) * (precision * recall) / ((5**2 * precision) + recall)
    
    return precision, recall, f1, f5

# You'll need to preprocess your predictions and true_labels to filter out padding based on attention masks and flatten them for sklearn metrics
# true_labels, predicted_labels should be flattened lists of labels for tokens where the corresponding attention_mask is 1
labels = list(range(1, 15))
precision, recall, f1, f5 = calculate_metrics(filtered_true_labels, filtered_predicted_labels, labels)
# Print the scores for each label
for i, label_id in enumerate(labels):
    print(f"Label {label_id} - Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1[i]}, F5: {f5[i]}")

# Calculate unweighted (macro) averages
average_precision = np.mean(precision)
average_recall = np.mean(recall)
average_f1 = np.mean(f1)
average_f5 = np.mean(f5)

print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1: {average_f1}")
print(f"Average F5: {average_f5}")