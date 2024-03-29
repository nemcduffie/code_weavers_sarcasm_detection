import pandas as pd
import re, string, nltk, json, argparse, os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from data.abbr_data import abbreviations

nltk.download('punkt')
nltk.download('stopwords')

DATA_DIR = os.path.join(os.getcwd(), 'data')


# Load data
def load_csv(file_path):
    return pd.read_csv(file_path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocess text data with emojis'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['train', 'test'],
        help='Specify the dataset file (train or test)',
    )
    return parser.parse_args()


def preprocess_text_with_emojis(text, emoji_to_text):
    # Remove usernames
    text = re.sub(r'@\S+', '', text)

    # Remove links
    text = re.sub(r'https?://\S+', '', text)

    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d+', '', text)

    # Insert space between adjacent emojis
    emoji_pattern = re.compile(r'([^\w\s])\1*')
    text = emoji_pattern.sub(r' \1 ', text)

    # Tokenize
    token_list = word_tokenize(text)

    # Replace emojis with their corresponding text labels
    token_list = [
        (
            emoji_to_text[token]
            if token in emoji_to_text and not token.startswith('skin_tone')
            else token
        )
        for token in token_list
    ]

    # Replace abbreviations with their full forms
    token_list = [
        abbreviations.get(token.lower(), token) for token in token_list
    ]

    # Remove non-textual tokens
    token_list = [
        token for token in token_list if any(c.isalpha() for c in token)
    ]

    # Remove stop words
    stop_word_set = set(stopwords.words('english'))
    token_list = [token for token in token_list if token not in stop_word_set]

    return token_list


def preprocess_dataset_with_emojis(dataset, emoji_to_text, target_columns):
    preprocessed_dataset = []

    for index, row in dataset.iterrows():
        if pd.isnull(row[target_columns[0]]):
            continue

        preprocessed_entry = {col: row[col] for col in target_columns}

        preprocessed_entry['text_with_emojis'] = preprocess_text_with_emojis(
            row[target_columns[0]], emoji_to_text
        )
        preprocessed_dataset.append(preprocessed_entry)

    return preprocessed_dataset


def apply_pretrained_embeddings(dataset, vocab_dict):
    for entry in dataset:
        entry['text_with_embeddings'] = [
            vocab_dict.get(token, [0]) for token in entry['text_with_emojis']
        ]
    return dataset


def load_embedding(embedding_file):
    vocab_dict = {}
    with open(embedding_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            vocab_dict[word] = vector
    return vocab_dict


def main(dataset):

    # Switch between train and test dataset and columns
    if dataset == 'train':
        dataset_file = os.path.join(DATA_DIR, 'train', 'train.En.csv')
        target_columns = [
            'tweet',
            'sarcastic',
            'rephrase',
            'sarcasm',
            'irony',
            'satire',
            'understatement',
            'overstatement',
            'rhetorical_question',
        ]
        output_file_path = os.path.join(DATA_DIR, 'prep_train.json')

    elif dataset == 'test':
        dataset_file = os.path.join(DATA_DIR, 'test', 'task_A_En_test.csv')
        target_columns = ['text', 'sarcastic']
        output_file_path = os.path.join(DATA_DIR, 'prep_test.json')

    else:
        print(
            "Invalid dataset argument. Please use --dataset train or --dataset test."
        )
        exit(1)

    # Load data
    text_dataset = load_csv(dataset_file)
    emoji_dataset = load_csv(
        os.path.join(DATA_DIR, "emoji_and_abbr/emoji_df.csv")
    )

    # Map emojis to text labels
    emoji_to_text = dict(zip(emoji_dataset['emoji'], emoji_dataset['name']))

    # Preprocess dataset with emojis
    preprocessed_data_with_emojis = preprocess_dataset_with_emojis(
        text_dataset, emoji_to_text, target_columns
    )

    # Save output file
    output_file_with_emojis_path = os.path.join(DATA_DIR, 'output_file.txt')
    with open(
        output_file_with_emojis_path, 'w', encoding='utf-8'
    ) as output_file_with_emojis:
        for entry in preprocessed_data_with_emojis:
            print(entry, file=output_file_with_emojis)

    print(f"Output saved to {output_file_with_emojis_path}")

    # Apply pretrained embeddings
    embedding_file = './data/glove.twitter.27B/glove.twitter.27B.200d.txt'
    vocab_dict = load_embedding(embedding_file)
    preprocessed_data_with_embeddings = apply_pretrained_embeddings(
        preprocessed_data_with_emojis, vocab_dict
    )

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(
            preprocessed_data_with_embeddings, output_file, ensure_ascii=False
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(args.dataset)
