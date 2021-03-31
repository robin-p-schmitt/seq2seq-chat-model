from ..dataset import ChatDataset
from pathlib import Path
import os
import string
import re


def test_tokens(dataset):
    punct = string.punctuation
    vocab = dataset.vocab
    allowed = {"<start>", "<stop>", "<pad>", "<new>", "<number>", "<unk>"}
    not_allowed = set(
        [
            token
            for token in vocab
            if re.search("[" + punct + "]", token) and len(token) > 1
        ]
    )
    not_allowed = not_allowed - allowed

    print(not_allowed)


if __name__ == "__main__":
    data_path = os.path.join("data", "training")
    file_paths = Path(data_path).glob("*.txt")
    dataset = ChatDataset(file_paths)

    test_tokens(dataset)
