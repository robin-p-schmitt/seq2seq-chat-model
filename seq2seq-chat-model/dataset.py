import torch
from torch.utils.data import Dataset
from pathlib import Path
import re
from tqdm import tqdm
import nltk
import gensim
import os


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

    def __init__(self, directory, max_length=10, embedding_size=128):
        """Initialize Dataset

        Get all sequences (questions and answers) from a directory of text
        files, create vocabulary and pretrain word2vec embeddings.
        """
        self.max_length = max_length
        self.embedding_size = embedding_size

        print("READING TXT FILES")
        self.data = self._get_data(directory)
        print("OBTAINING EMBEDDINGS AND VOCABULARY")
        self.vocab, self.word2vec, self.sentences = self._get_vocab(self.data)
        self.inverse_vocab = {val: key for key, val in self.vocab.items()}
        print("DATA LOADED")

    def _get_data(self, directory):
        """Get tuples of questions and answers from directory of txt
        files."""
        data = []
        for filename in tqdm(list(Path(directory).glob("*.txt"))):
            sequences = self._get_sequences(filename)
            data += sequences

        return data

    def _get_sequences(self, filename):
        """Get tuples of questions and answers from specific txt file."""
        sequences = []

        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                # each line consists of
                # [question_1, ...]\t[answer_1, ...]\n
                split = re.findall("(.+?)\t(.+?)\n", line)
                if split:
                    question_group, answer_group = split[0]

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
                        self._replace_digits(token)
                        for token in nltk.word_tokenize(question)
                    ]
                    for question in question_group
                ]
                answer_group = [
                    [
                        self._replace_digits(token)
                        for token in nltk.word_tokenize(answer)
                    ]
                    for answer in answer_group
                ]

                sequences.append((question_group, answer_group))

        return sequences

    def _replace_digits(self, txt):
        if re.fullmatch(r"\d+", txt):
            return "<number>"

        return txt

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

        word2vec = gensim.models.Word2Vec(min_count = 5, size = 128)
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

    def get_embeddings(self):
        """Return the pretrained embeddings together with 0-initialized
        embeddings for the 5 special tokens"""
        embeddings = torch.cat(
            (
                torch.zeros((5, self.embedding_size)),
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
        # link together sequences of one group with the <new> token e.g.
        # question_group = [["hey"], ["how", "are", "you", "?"]] ->
        # question = ["hey", "<new>", "how", "are", "you", "?"]
        question, answer = [
            [token for seq in seq_group for token in seq + ["<new>"]]
            for seq_group in [question_group, answer_group]
        ]
        # replace every token with its unique index or with the <unk> index
        # if it is not in the vocabulary
        question, answer = [
            [
                self.vocab[token]
                if token in self.vocab
                else self.vocab["<unk>"]
                for token in seq
            ]
            for seq in [question, answer]
        ]

        # either cut off long sequences or pad short sequences so that
        # every sequence has length max_length
        question = question[: self.max_length] + [self.vocab["<pad>"]] * max(
            self.max_length - len(question), 0
        )
        # additionally, add sos and eos tokens to start and end of the
        # answer
        answer = (
            [self.vocab["<start>"]]
            + answer[: self.max_length - 2]
            + [self.vocab["<stop>"]]
            + [self.vocab["<pad>"]] * max(self.max_length - len(answer) - 2, 0)
        )

        return (torch.tensor(question), torch.tensor(answer))
