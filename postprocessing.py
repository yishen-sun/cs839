from transformers import TFBertForTokenClassification
import numpy as np
import pandas as pd
import tensorflow as tf
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

all_predicted_labels = np.argmax(predictions.logits, axis=-1)
all_true_labels = []
all_attention_masks = []
for batch in test_dataset.unbatch():  # Iterate over each batch in the dataset
    all_true_labels.append(batch[1].numpy())  # Assuming the second element of the batch is numeric_labels
    all_attention_masks.append(batch[0]['attention_mask'].numpy())  # Assuming attention_mask is part of the input dict

data = {
    'all_predicted_labels': all_predicted_labels,
    'all_true_labels': all_true_labels,
    'all_attention_masks': all_attention_masks
}
data_df = pd.DataFrame(data)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
data_df.to_parquet(f'predicted_results_{timestamp}.parquet', engine='pyarrow')
