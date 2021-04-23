from seq2seq_chat_model.dataset import ChatDataset
import os
import unittest
import gensim
import random
import string


class TestChatDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.dir, "data")
        self.data_path = os.path.join(self.data_dir, "test.txt")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

            with open(self.data_path, "w+") as f:
                for _ in range(10):
                    str1 = random.choices(string.ascii_letters, k=10)
                    str2 = random.choices(string.ascii_letters, k=10)

                    f.write(
                        str([" ".join(str1)])
                        + "\t"
                        + str([" ".join(str2)])
                        + "\n"
                    )

        self.max_length = 15

        self.emb_size = 128

    def test_dataset_init(self):
        ChatDataset([self.data_path], max_length=self.max_length)

    def test_sequences(self):
        dataset = ChatDataset([self.data_path], max_length=self.max_length)

        question, answer = dataset[0]

        self.assertSequenceEqual(question.shape, [self.max_length])
        self.assertSequenceEqual(answer.shape, [self.max_length])

    def test_embeddings(self):
        dataset = ChatDataset([self.data_path], max_length=self.max_length)

        sentences = dataset.sentences

        word2vec = gensim.models.Word2Vec(sentences, size=self.emb_size)

        dataset.load_word2vec(word2vec)

        embeddings = dataset.get_embeddings()

        self.assertSequenceEqual(
            embeddings.shape, [len(dataset.vocab), self.emb_size]
        )


if __name__ == "__main__":
    unittest.main()
