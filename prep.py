import pandas as pd
import random, sys, re, string, nltk, json, argparse, os

from alive_progress import alive_bar
from alive_progress import config_handler
from data.abbr_data import abbreviations
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

config_handler.set_global(length=50, file=sys.stderr)

nltk.download('punkt')
nltk.download('stopwords')

DATA_DIR = os.path.join(os.getcwd(), 'data')
PRUNE_DATA = os.getenv('PRUNE_DATA', '0') == '1'


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
        choices=['train', 'train_extra', 'test'],
        help='Specify the dataset file (train, train_extra, or test)',
    )
    return parser.parse_args()


def preprocess_text_with_emojis(text, emoji_to_text):
    # Remove usernames
    text = re.sub(r'@\S+', '', text)

    # Remove links
    text = re.sub(r'https?://\S+', '', text)

    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
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


def preprocess_dataset_with_emojis(
    dataset_file, emoji_to_text, target_columns, vocab
):
    preprocessed_dataset = []
    dataset = load_csv(dataset_file)
    with alive_bar(len(dataset), bar='circles', title='Process emojis ') as bar:
        for _, row in dataset.iterrows():
            bar()
            if row.get('text_0', False):
                row = {
                    'tweet': row.get(f'text_{row["sarcastic_id"]}'),
                    'sarcastic': 1,
                    'rephrase': row.get(f'text_{abs(row["sarcastic_id"]-1)}'),
                }
            elif pd.isnull(row[target_columns[0]]):
                continue

            preprocessed_entry = {col: row.get(col) for col in target_columns}

            preprocessed_entry['text_with_emojis'] = (
                preprocess_text_with_emojis(
                    row[target_columns[0]], emoji_to_text
                )
            )
            for word in preprocessed_entry['text_with_emojis']:
                vocab.add(word)

            preprocessed_dataset.append(preprocessed_entry)
    return preprocessed_dataset, vocab


def apply_pretrained_embeddings(dataset, vocab_dict):
    with alive_bar(
        len(dataset), bar='brackets', title='Apply embedding'
    ) as bar:
        for entry in dataset:
            entry['text_with_embeddings'] = [
                vocab_dict.get(token, [0])
                for token in entry['text_with_emojis']
            ]
            bar()
    return dataset


def load_embedding(embedding_file, vocab):
    vocab_dict = {}
    with alive_bar(
        1193514, bar='filling', title='Load embedding '
    ) as bar:  # Embedding file has 1193514 lines
        with open(embedding_file, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                if word in vocab:
                    vector = [float(val) for val in values[1:]]
                    vocab_dict[word] = vector
                bar()
    return vocab_dict


def main(dataset):
    # Switch between train and test dataset and columns
    if dataset in ['train', 'train_extra']:
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
            'Invalid dataset argument. Please use --dataset train, --dataset train_extra, or --dataset test.'
        )
        exit(1)

    # Load emoji data
    emoji_dataset = load_csv(
        os.path.join(DATA_DIR, 'emoji_and_abbr/emoji_df.csv')
    )

    # Map emojis to text labels
    emoji_to_text = dict(zip(emoji_dataset['emoji'], emoji_dataset['name']))

    vocab = set([])
    # Preprocess dataset with emojis
    preprocessed_data, vocab = preprocess_dataset_with_emojis(
        dataset_file, emoji_to_text, target_columns, vocab
    )

    # Optional adaptation of task_C_En_test.csv
    # (originally intended as test data for another task)
    if dataset == 'train_extra':
        original_len_data = len(preprocessed_data)
        extra_dataset_file = os.path.join(
            DATA_DIR, 'test', 'task_C_En_test.csv'
        )
        extra_preprocessed, vocab = preprocess_dataset_with_emojis(
            extra_dataset_file,
            emoji_to_text,
            target_columns,
            vocab,
        )

        preprocessed_data.extend(extra_preprocessed)
        print(
            f'Extra dataset increases training data from {original_len_data} to {len(preprocessed_data)} entries ({len(extra_preprocessed)} additional sarcastic entries).'
        )

    # Save output file
    output_file_with_emojis_path = os.path.join(DATA_DIR, 'output_file.txt')
    with open(
        output_file_with_emojis_path, 'w', encoding='utf-8'
    ) as output_file_with_emojis:
        for entry in preprocessed_data:
            print(entry, file=output_file_with_emojis)

    print(f'Output saved to {output_file_with_emojis_path}')

    # Apply pretrained embeddings
    embedding_file = './data/glove.twitter.27B/glove.twitter.27B.200d.txt'
    vocab_dict = load_embedding(embedding_file, vocab)
    preprocessed_data_with_embeddings = apply_pretrained_embeddings(
        preprocessed_data, vocab_dict
    )

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(
            preprocessed_data_with_embeddings, output_file, ensure_ascii=False
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(args.dataset)
