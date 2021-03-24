### Seq2Seq Chat System

This repository contains two notebooks which can be used to train a seq2seq chat model.

The file "prepare_data.ipynb" converts data from different sources into a common format to use as training data.
The file "chat_model.ipynb" contains functions to load the training data, train a seq2seq model and let the trained model produce answers for given input sentences.

#### Next Steps

- pretrain the model on the cornell movie corpus (which is in English) and see if it helps the performance of the model when it is being fine-tuned on German WhatsApp chats. 