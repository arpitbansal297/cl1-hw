import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time
import nltk

kUNK = '<unk>'
kPAD = '<pad>'


# You don't need to change this funtion
def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i += 1
    return class_to_i, i_to_class


# You don't need to change this funtion
def load_data(filename, lim):
    """
    load the json file into data list
    """

    data = list()
    with open(filename) as json_data:
        if lim > 0:
            questions = json.load(json_data)[:lim]
        else:
            questions = json.load(json_data)
        for q in questions:
            q_text = nltk.word_tokenize(q['question_text'])
            # label = q['category']
            label = q['label']

            if label is not None:
                data.append((q_text, label))
    return data


# You don't need to change this funtion
def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {kPAD: 0, kUNK: 1}
    ind2word = {0: kPAD, 1: kUNK}
    for q_text, _ in exs:
        for w in q_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word


class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples, word2ind, num_classes, class2ind=None):
        self.questions = []
        self.labels = []

        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)

        if type(self.labels[0]) == str:
            for i in range(len(self.labels)):
                try:
                    self.labels[i] = class2ind[self.labels[i]]
                except:
                    self.labels[i] = num_classes
        self.word2ind = word2ind

    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.vectorize(self.questions[index], self.word2ind), \
               self.labels[index]

    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(ex, word2ind):
        """
        vectorize a single example based on the word2ind dict.
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        vec_text = [0] * len(ex)
        for i in range(len(ex)):
            if ex[i] in word2ind.keys():
                vec_text[i] = word2ind[ex[i]]
            else:
                vec_text[i] = word2ind['<unk>']

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized
        ####Your code here

        return vec_text


###You don't need to change this funtion

def batchify(batch):
    """
    Gather a batch of individual examples into one batch,
    which includes the question text, question length and labels
    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch

def evaluate_analysis(data_loader, model, device, ind2word):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        logits = model(question_text, question_len, True)
        ####Your code here

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)

        if top_i != labels:
            str = ''
            question_text = question_text.cpu().detach().numpy()
            for ind in question_text[0]:
                str = str + ind2word[ind] + " "
            print(str)
            print("Actual Label : ", labels[0])
            print("Predicted Label : ", top_i[0])

    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy

def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        logits = model(question_text, question_len, True)
        ####Your code here

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()
    cnt = 0

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        optimizer.zero_grad()
        output = model(question_text, question_len, True)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        #### Your code here

        clip_grad_norm_(model.parameters(), args.grad_clipping)
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()
        cnt+=1

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time() - start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy

    # print("Loss of epoch : ", epoch_loss_total/cnt)
    return accuracy


class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model for passing tests, might need to tinker to improve performance/handle
    #### pretrained word embeddings/for your project code.

    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.dropout = nn.Dropout(nn_dropout)
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)
        self._softmax = nn.Softmax()

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU.
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        #### Your code here

    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text : vectorized question text
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        # logits = torch.LongTensor([0.0] * self.n_classes)
        out = self.embeddings(input_text)

        out = out.sum(1)
        out /= text_len.view(out.size(0), -1)

        out = F.relu(self.linear1(out))
        out = self.dropout(out)
        logits = self.linear2(out)

        # Complete the forward funtion.  First look up the word embeddings.

        # Then average them

        # Before feeding them through the network

        if is_prob:
            logits = self._softmax(logits)

        return logits


# You basically do not need to modify the below code
# But you may need to add funtions to support error analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--train-file', type=str, default='data/question_train_cl1.json')
    parser.add_argument('--dev-file', type=str, default='data/question_dev_cl1.json')
    parser.add_argument('--test-file', type=str, default='data/question_test_cl1.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    ### Load data
    train_exs = load_data(args.train_file, args.limit)
    dev_exs = load_data(args.dev_file, -1)
    test_exs = load_data(args.test_file, -1)

    ### Create vocab
    voc, word2ind, ind2word = load_words(train_exs)

    # get num_classes from training + dev examples - this can then also be used as int value for those test class labels not seen in training+dev.
    num_classes = len(list(set([ex[1] for ex in train_exs + dev_exs])))
    print(num_classes)

    # get class to int mapping
    class2ind, ind2class = class_labels(train_exs + dev_exs)

    if args.test:
        model = torch.load(args.load_model)
        #### Load batchifed dataset
        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler, num_workers=0,
                                                  collate_fn=batchify)
        evaluate(test_loader, model, device)
    else:
        if args.resume:
            model = torch.load(args.load_model)
        else:
            model = DanModel(num_classes, len(voc))
            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = QuestionDataset(train_exs, word2ind, num_classes, class2ind)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                                 sampler=dev_sampler, num_workers=0,
                                                 collate_fn=batchify)
        accuracy = 0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       sampler=train_sampler, num_workers=0,
                                                       collate_fn=batchify)
            accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
            # print("Accuracy : ", accuracy)
        print('start testing:\n')

        print("On Test dataset")

        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  sampler=test_sampler, num_workers=0,
                                                  collate_fn=batchify)
        evaluate(test_loader, model, device)


        # # Analysis
        # dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
        # dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        # dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=1,
        #                                          sampler=dev_sampler, num_workers=0,
        #                                          collate_fn=batchify)
        #
        # evaluate_analysis(dev_loader, model, device, ind2word)