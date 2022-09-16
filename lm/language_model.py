from math import log, exp
from collections import defaultdict
import argparse

from numpy import mean

import nltk
from nltk import FreqDist
from nltk.util import bigrams
from nltk.tokenize import TreebankWordTokenizer

kLM_ORDER = 2
kUNK_CUTOFF = 3
kNEG_INF = -1e6

kSTART = "<s>"
kEND = "</s>"

def lg(x):
    return log(x) / log(2.0)

class BigramLanguageModel:

    def __init__(self, unk_cutoff, jm_lambda=0.6, dirichlet_alpha=0.1,
                 katz_cutoff=5, kn_discount=0.1, kn_concentration=1.0,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lambda x: x.lower()):
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._dirichlet_alpha = dirichlet_alpha
        self._katz_cutoff = katz_cutoff
        self._kn_concentration = kn_concentration
        self._kn_discount = kn_discount
        self._vocab_final = False

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function
        self.vocab = {}
        self.vocab[kSTART] = (self._unk_cutoff+1, 0)
        self.vocab[kEND] = (self._unk_cutoff+1, 1)
        self.cut_counts = 2 # it will tell how many things in vocab which are unique

        self.unigram = {}
        self.unigram_slides = {}
        self.unigram_slides['total_number_to_divide'] = 0
        self.bigram = {}
        
        # Add your code here!

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"
        if word in self.vocab.keys():
            self.vocab[word] = (self.vocab[word][0] + 1, self.vocab[word][1])
        else:
            self.vocab[word] = (count, -1)

        if self.vocab[word][0] > self._unk_cutoff:
            self.vocab[word] = (self.vocab[word][0], self.cut_counts)
            self.cut_counts += 1

        # Add your code here!            

    def tokenize(self, sent):
        """
        Returns a generator over tokens in the sentence.  

        You don't need to modify this code.
        """
        for ii in self._tokenizer(sent):
            yield ii
        
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"
        if word in self.vocab:
            return self.vocab[word][1]
        # Add your code here
        return -1

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # You probably do not need to modify this code
        self._vocab_final = True

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or
        testing.  Prefix the sentence with <s>, replace words not in
        the vocabulary with <UNK>, and end the sentence with </s>.

        You should not modify this code.
        """
        yield self.vocab_lookup(kSTART)
        for ii in self._tokenizer(sentence):
            yield self.vocab_lookup(self._normalizer(ii))
        yield self.vocab_lookup(kEND)


    def normalize(self, word):
        """
        Normalize a word

        You should not modify this code.
        """
        return self._normalizer(word)


    def mle(self, context, word):
        """
        Return the log MLE estimate of a word given a context.  If the
        MLE would be negative infinity, use kNEG_INF
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        dict = self.bigram[context]
        if word in dict.keys():
            val = dict[word] / dict['total_number_to_divide']
            return lg(val)

        return kNEG_INF

    def laplace(self, context, word):
        """
        Return the log MLE estimate of a word given a context.
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        dict = self.bigram[context]
        if word in dict.keys():
            val = dict[word]
        else:
            val = 0

        ans = (1 + val) / (dict['total_number_to_divide'] + len(self.unigram_slides.keys()))
        return lg(ans)

    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        if word in self.bigram[context].keys():
            binary_prob = self.bigram[context][word] / self.bigram[context]['total_number_to_divide']
        else:
            binary_prob = 0
        unary_prob = self.unigram_slides[word] / self.unigram_slides['total_number_to_divide']
        return lg(self._jm_lambda * binary_prob + (1 - self._jm_lambda) * unary_prob)

    def kneser_ney(self, context, word):
        """
        Return the log probability of a word given a context given
        Kneser Ney backoff
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        theta = self._kn_concentration
        delta = self._kn_discount
        if word in self.bigram[context].keys():
            c_u_x = self.bigram[context][word]
        else:
            c_u_x = 0.0

        c_u = self.bigram[context]['total_number_to_divide']

        val1 = (c_u_x - delta) / (theta + c_u)
        val2 = (theta + delta * (len(self.bigram[context])-1)) / (theta + c_u)

        c_phi_x = self.unigram_slides[word]
        c_phi = self.unigram_slides['total_number_to_divide']

        val3 = (c_phi_x - delta) / (theta + c_phi)
        val4 = (theta + delta * (len(self.unigram_slides)-1)) / (theta + c_phi)
        val4 = val4/ (len(self.unigram_slides.keys()))

        val1 = max(0, val1)
        val2 = max(0, val2)
        val3 = max(0, val3)
        val4 = max(0, val4)

        total = val1 + val2 * (val3 + val4)
        return lg(total)

    def dirichlet(self, context, word):
        """
        Additive smoothing, assuming independent Dirichlets with fixed
        hyperparameter.
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        cons = self._dirichlet_alpha
        dict = self.bigram[context]
        if word in dict.keys():
            val = dict[word]
        else:
            val = 0

        ans = (cons + val) / (dict['total_number_to_divide'] + ( len(self.unigram_slides.keys()) )*cons)
        return lg(ans)

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of
        # code that will hopefully get you started.
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            key = context
            if key in self.bigram.keys():
                key_dict = self.bigram[key]
                if word in key_dict.keys():
                    key_dict[word] += 1
                else:
                    key_dict[word] = 1
                    # to back off since nothing was there
                    new_key = word
                    if new_key not in self.unigram_slides.keys():
                        self.unigram_slides[new_key] = 1
                    else:
                        self.unigram_slides[new_key] += 1

                    self.unigram_slides['total_number_to_divide'] += 1



                self.bigram[key] = key_dict
                self.bigram[key]['total_number_to_divide'] += 1

            else:
                self.bigram[key] = {}
                self.bigram[key]['total_number_to_divide'] = 1
                self.bigram[key][word] = 1

                new_key = word
                if new_key not in self.unigram_slides.keys():
                    self.unigram_slides[new_key] = 1
                else:
                    self.unigram_slides[new_key] += 1

                self.unigram_slides['total_number_to_divide'] += 1


        # print(self.bigram)
        # print(self.unigram_slides)

    def perplexity(self, sentence, method):
        """
        Compute the perplexity of a sentence given a estimation method

        You do not need to modify this code.
        """
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.tokenize_and_censor(sentence))]))

    def sample(self, method, samples=25):
        """
        Sample words from the language model.
        
        @arg samples The number of samples to return.
        """
        # Modify this code to get extra credit.  This should be
        # written as an iterator.  I.e. yield @samples times followed
        # by a final return, as in the sample code.

        for ii in xrange(samples):
            yield ""
        return

# You do not need to modify the below code, but you may want to during
# your "exploration" of high / low probability sentences.
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--dir_alpha", help="Dirichlet parameter " + \
                           "for pseudocounts",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--katz_cutoff", help="Cutoff when to use Katz " + \
                           "backoff",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--brown_limit", help="How many sentences to add " + \
                           "from Brown",
                           type=int, default=-1, required=False)
    argparser.add_argument("--kn_discount", help="Kneser-Ney discount parameter",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--kn_concentration", help="Kneser-Ney concentration parameter",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--method", help="Which LM method we use",
                           type=str, default='laplace', required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             dirichlet_alpha=args.dir_alpha,
                             katz_cutoff=args.katz_cutoff,
                             kn_concentration=args.kn_concentration,
                             kn_discount=args.kn_discount)

    for ii in nltk.corpus.brown.sents():
        for jj in lm.tokenize(" ".join(ii)):
            lm.train_seen(lm._normalizer(jj))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    sentence_count = 0
    for ii in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(" ".join(ii))

        if args.brown_limit > 0 and sentence_count >= args.brown_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)
    assert args.method in ['kneser_ney', 'mle', 'dirichlet', \
                           'jelinek_mercer', 'good_turing', 'laplace'], \
      "Invalid estimation method"

    sent = input()
    while sent:
        print("#".join(str(x) for x in lm.tokenize_and_censor(sent)))
        print(lm.perplexity(sent, getattr(lm, args.method)))
        sent = input()

    # stats = {}
    # sentence_count = 0
    # for ii in nltk.corpus.treebank.sents():
    #     sent = " ".join(ii)
    #
    #     indexes = "#".join(str(x) for x in lm.tokenize_and_censor(sent))
    #     perp = lm.perplexity(sent, getattr(lm, args.method))
    #     stats[perp] = (sent, indexes)
    #
    #
    # l = list(stats.keys())
    # l.sort()

    # cnt = 0
    # for ll in l:
    #     # print("############################################3")
    #     # print(ll)
    #     print(stats[ll][0])
    #     # print(stats[ll][1])
    #     if cnt == 10:
    #         break
    #
    #     cnt += 1


