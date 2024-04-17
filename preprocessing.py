import tensorflow as tf
import pandas as pd
import json
from transformers import BertTokenizerFast
import tqdm

# Load the datasets
train_data_path = './train.json'
test_data_path = './test.json'

# Load the JSON data from the file
with open(train_data_path, 'r') as file:
    data = json.load(file)
    # Convert the loaded JSON data into a pandas DataFrame
    train_df = pd.DataFrame(data)
with open(test_data_path, 'r') as file:
    data = json.load(file)
    # Convert the loaded JSON data into a pandas DataFrame
    test_df = pd.DataFrame(data)
# Quick look at the data structure
# print(train_df.head())
# print(test_df.head())

pii_types = [label for sublist in train_df['labels'].tolist() for label in sublist]
pii_types_df = pd.DataFrame(pii_types, columns=['PII_Type'])

# View the distribution of PII types
# print(pii_types_df['PII_Type'].value_counts())

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

def segment_text_by_tokens(original_tokens, original_labels, max_length, overlap):
    # Adjust max_length to account for special tokens like [CLS] and [SEP] which are added by models like BERT
    max_length -= 2  # Assuming 2 special tokens: [CLS] and [SEP]

    segments = []
    current_tokens = []
    current_labels = []

    for token, label in zip(original_tokens, original_labels):
        # Check if adding the next token would exceed the max_length
        if len(current_tokens) >= max_length and current_tokens:
            # Finalize the current segment before the overflow
            segments.append({
                "text": ' '.join(current_tokens),
                "tokens": current_tokens,
                "labels": current_labels
            })
            # Start new segment with overlap if specified
            current_tokens = current_tokens[-overlap:] if overlap else []
            current_labels = current_labels[-overlap:] if overlap else []

        current_tokens.append(token)
        current_labels.append(label)
    
    # Add the last segment if there are remaining tokens
    if current_tokens:
        segments.append({
            "text": ' '.join(current_tokens),
            "tokens": current_tokens,
            "labels": current_labels
        })
    
    return segments

def get_segment_df(df, max_length=512, overlap=16):
    all_segments = []
    j = 0
    for index, row in df.iterrows():
        # original_text = row['full_text']
        original_tokens = row['tokens']
        original_labels = row['labels']
        segments = segment_text_by_tokens(original_tokens, original_labels, max_length, overlap)
        for i, segment in enumerate(segments):
            # print(f"Segment {j}:")
            j += 1
            # print(f"Text: {segment['text']}")
            # print(f"Tokens: {segment['tokens']}")
            # print(f"Labels: {segment['labels']}\n")
            for segment in segments:
                all_segments.append({
                    "text": segment['text'],
                    "tokens": segment['tokens'],
                    "labels": segment['labels']
                })
    # Convert the list of dictionaries to a DataFrame
    segments_df = pd.DataFrame(all_segments)
    return segments_df
    
# Write the DataFrame to a Parquet file
segments_df = get_segment_df(train_df, 441, 0)
segments_df.to_parquet('segments.parquet', engine='pyarrow')
segments_df.info()


tokenized_tokens = []
tokenized_labels = []
numeric_labels = [] # important
attention_masks = [] # important
tokenized_input_ids = [] # important
index_mappings = [] # important
# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(segment_text, segment_tokens, segment_labels):
    # Step 1: Generate Original Token Offset Mapping
    original_token_offsets = []
    cursor = 0
    for token in segment_tokens:
        start = segment_text.find(token, cursor)
        end = start + len(token)
        cursor = end
        original_token_offsets.append((start, end))
    
    # Create a mapping from original token offsets to labels
    offset_to_label = {}
    for offset, label in zip(original_token_offsets, segment_labels):
        offset_to_label[offset] = label

    # Step 2: Tokenize and Align Labels
    encoded = tokenizer(segment_text, return_offsets_mapping=True, truncation=True, padding='max_length')
    input_ids = encoded['input_ids']
    offset_mapping = encoded['offset_mapping']
    attention_mask = encoded['attention_mask']
    new_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    new_mapping = [-1] * len(new_tokens)
    # Prepare to align new tokens with original labels
    new_labels = ['O'] * len(new_tokens) # Initialize all labels as 'O'
    # Unpack and transform a list of tuples into two separate lists
    orig_starts, orig_ends = zip(*original_token_offsets) if original_token_offsets else ([], [])
    j = 0  # Pointer for original_token_offsets

    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens. [CLS], [SEP], [PAD] are all (0, 0)
            continue
        # Move the original pointer to the right place
        while j < len(orig_starts) and orig_ends[j] < start:
            j += 1
        # Check if the new token is within the span of the original token
        if j < len(orig_starts) and orig_starts[j] <= start and end <= orig_ends[j]:
            # Assign the original token's label to the new token
            base_label = offset_to_label[(orig_starts[j], orig_ends[j])]
            if orig_starts[j] == start:
                new_labels[i] = base_label
            else:
                # Handling subwords and continuation labels: Adjust the new_labels to use continuation labels ('I-') where necessary
                if base_label.startswith('B-'):
                    new_labels[i] = 'I-' + base_label[2:]
                else:
                    new_labels[i] = base_label
            new_mapping[i] = j

    # Step 3: Review the results
    # print(new_tokens)
    # print(new_labels)
    # Convert segment labels to numeric and align with tokens
    numeric_label = [label_map[label] for label in new_labels]
    return new_tokens, input_ids, new_labels, numeric_label, attention_mask, new_mapping


for i, segment in tqdm(segments_df.iterrows(), total=segments_df.shape[0]):
    segment_new_tokens, input_ids, segment_new_labels, numeric_label, segment_attention_mask, index_mapping = tokenize_and_align_labels(segment['text'], segment['tokens'], segment['labels'])
    tokenized_tokens.append(segment_new_tokens)
    tokenized_input_ids.append(input_ids)
    tokenized_labels.append(segment_new_labels)
    numeric_labels.append(numeric_label)
    attention_masks.append(segment_attention_mask)
    index_mappings.append(index_mapping)

preprocessed_data = {
    'numeric_labels': numeric_labels,
    'attention_masks': attention_masks,
    'input_ids': tokenized_input_ids,
    'index_mappings': index_mappings
}

preprocessed_data_df = pd.DataFrame(preprocessed_data)
preprocessed_data_df.to_parquet('preprocessed_data.parquet', engine='pyarrow')