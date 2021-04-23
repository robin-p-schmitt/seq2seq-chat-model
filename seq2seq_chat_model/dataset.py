import torch
from torch.utils.data import Dataset
import re
from tqdm import tqdm
import nltk
import gensim
from seq2seq_chat_model.data.prepare_data import replace_digits
from seq2seq_chat_model.models.utils import (
    get_encoder_input,
    get_decoder_input,
)


class ChatDataset(Dataset):
    """Dataset which contains tuples of questions and answers.

    This dataset extracts sequences from txt files. The txt files need to
    store one pair of question and answer per line in the following form:

    "['question_1','question_2',...]\t['answer_1','answer_2',...]\n"

    Each line contains a list of questions and answers, because a question
    can consist of multiple messages, e.g. in a WhatsApp chat one user
    might send multiple consecutive messages without an answer in between.

    The dataset returns tuples of questions and answers, where questions
    and answers are given as lists of unique word indices. Multiple
    consecutive questions and answers are linked using the <new> token to
    symbolize the start of a new message.


    Functions: __get_data: loads data from a directory with txt files.
        __get_sequences: loads tuples of questions and answers from
        specific txt file. __get_vocab: creates vocabulary and pretrains
        word2vec model for the loaded sequences. get_embeddigns: returns
        the pretrained embeddigns of the word2vec model.

    Attributes: max_length: the maximum length of a sequence.
        embedding_size: the size of the pretrained embeddings. vocab: maps
        tokens to unique indices. inverse_vocab: maps indices to tokens.
        word2vec: trained gensim word2vec model.
    """

    def __init__(self, file_paths, max_length=10, min_count=5):
        """Initialize Dataset

        Get all sequences (questions and answers) from a directory of text
        files, create vocabulary and pretrain word2vec embeddings.

        Args:
            file_paths (Iterable): an iterable of file paths to get the data
                                    from.
            max_length (int): the maximum length of returned sequences.
            embedding_size (int): size of word embeddings.

        """
        self.max_length = max_length
        self.min_count = min_count

        print("READING TXT FILES")
        self.data = self._get_data(file_paths)
        print("OBTAINING VOCABULARY")
        self.vocab, self.word2vec, self.sentences = self._get_vocab(self.data)
        self.inverse_vocab = {val: key for key, val in self.vocab.items()}
        print("DATA LOADED")

    def _get_data(self, file_paths):
        """Get tuples of questions and answers from directory of txt
        files."""
        data = []
        for path in tqdm(list(file_paths)):
            sequences = self._get_sequences(path)
            data += sequences

        if len(data) == 0:
            raise ValueError(
                "No data could be extracted from the given files. Try checking the formatting of the files."
            )

        return data

    def _get_sequences(self, path):
        """Get tuples of questions and answers from specific txt file."""
        sequences = []

        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                # each line consists of
                # [question_1, ...]\t[answer_1, ...]\n
                split = re.findall("(.+?)\t(.+?)\n", line)
                if split:
                    question_group, answer_group = split[0]
                else:
                    continue

                # parse the strings to Python lists, e.g. question_group =
                # ["hey", "how are you?"] and answer_group = ["good", "what
                # about you?"]
                question_group, answer_group = eval(question_group), eval(
                    answer_group
                )
                # tokenize each message, e.g. question_group = [["hey"],
                # ["how", "are", "you", "?"]]
                question_group = [
                    [
                        replace_digits(token)
                        for token in nltk.word_tokenize(question)
                    ]
                    for question in question_group
                ]
                answer_group = [
                    [
                        replace_digits(token)
                        for token in nltk.word_tokenize(answer)
                    ]
                    for answer in answer_group
                ]

                # remove sequences which exceed the max length
                if (
                    len(
                        [
                            token
                            for question in question_group
                            for token in question
                        ]
                    )
                    + len(question_group)
                    > self.max_length
                ):
                    continue
                if (
                    len([token for answer in answer_group for token in answer])
                    + len(answer_group)
                    > self.max_length
                ):
                    continue

                sequences.append((question_group, answer_group))

        return sequences

    def _get_vocab(self, data):
        # unzip to obtain question groups and answer groups
        question_groups, answer_groups = list(zip(*data))

        # gather all sequences from all groups into a single list of
        # sentences
        questions = [
            question
            for question_group in question_groups
            for question in question_group
        ]
        answers = [
            answer for answer_group in answer_groups for answer in answer_group
        ]
        sentences = questions + answers

        word2vec = gensim.models.Word2Vec(min_count=5)
        word2vec.build_vocab(sentences)

        # obtain vocabulary including special tokens
        vocab = {
            token: index + 5
            for index, token in enumerate(word2vec.wv.index2word)
        }
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab["<start>"] = 2
        vocab["<stop>"] = 3
        vocab["<new>"] = 4

        return vocab, word2vec, sentences

    def load_word2vec(self, model):
        # obtain vocabulary including special tokens
        vocab = {
            token: index + 5 for index, token in enumerate(model.wv.index2word)
        }
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab["<start>"] = 2
        vocab["<stop>"] = 3
        vocab["<new>"] = 4

        self.word2vec = model
        self.vocab = vocab

    def get_embeddings(self):
        """Return the pretrained embeddings together with 0-initialized
        embeddings for the 5 special tokens"""
        size = self.word2vec.wv.vectors.shape[-1]
        embeddings = torch.cat(
            (
                torch.zeros((5, size)),
                torch.FloatTensor(self.word2vec.wv.vectors),
            )
        )
        return embeddings

    def __len__(self):
        """Return size of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """Obtain tuple at given index"""
        # print(self.data[index])
        question_group, answer_group = self.data[index]

        question = get_encoder_input(
            question_group,
            self.vocab,
            self.max_length,
            "<new>",
            "<unk>",
            "<pad>",
        )
        answer = get_decoder_input(
            answer_group,
            self.vocab,
            self.max_length,
            "<new>",
            "<unk>",
            "<pad>",
            "<start>",
            "<stop>",
        )

        return (torch.LongTensor(question), torch.LongTensor(answer))


class MockDataset(Dataset):
    """A dataset for testing purposes."""

    def __init__(self) -> None:
        self.vocab = {i: i for i in range(10)}
        self.x = torch.LongTensor(torch.randint(0, 10, (100, 10)))
        self.y = torch.LongTensor(torch.randint(0, 10, (100, 10)))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
