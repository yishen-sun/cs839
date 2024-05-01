from transformers import TFBertForTokenClassification
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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

all_predicted_labels = []
all_true_labels = []
all_attention_masks = []
all_predicted_labels = np.argmax(predictions['logits'], axis=-1).tolist()
i = 0
for (input_dict, true_labels) in test_dataset.unbatch().as_numpy_iterator():
    all_true_labels.append(true_labels.tolist())  # Appends the true labels of the batch
    all_attention_masks.append(input_dict['attention_mask'].tolist())  # Appends the attention masks of the batch
data = {
    'all_predicted_labels': all_predicted_labels,
    'all_true_labels': all_true_labels,
    'all_attention_masks': all_attention_masks
}
# Print the length of each sublist to check consistency
print(len(all_predicted_labels))
print(len(all_true_labels))
print(len(all_attention_masks))
for i, (pl, tl, am) in enumerate(zip(all_predicted_labels, all_true_labels, all_attention_masks)):
    if len(pl) != 512 or len(tl) != 512 or len(am) != 512:
        print(f"Row {i} lengths -- Predicted: {len(pl)}, True: {len(tl)}, Masks: {len(am)}")

data_df = pd.DataFrame(data)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
data_df.to_parquet(f'predicted_results_{timestamp}.parquet', engine='pyarrow')
