# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
from string import punctuation

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize


class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        dictionary = nltk.corpus.cmudict.dict()
        if word not in dictionary.keys():
            return 1

        split = dictionary[word][0] # just take one
        cnt = 0
        for s in split:
            if s[-1] in ['0', '1', '2']:
                cnt+=1

        return cnt

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        dictionary = nltk.corpus.cmudict.dict()
        if a not in dictionary.keys():
            return False
        if b not in dictionary.keys():
            return False

        num_a = self.num_syllables(a)
        num_b = self.num_syllables(b)

        min_num = min(num_a, num_b)

        splits_a = dictionary[a]
        splits_b = dictionary[b]

        new_splits_a = []
        for split_a in splits_a:
            total_length = len(split_a)
            new_split = []
            word_length = 0
            for i in range(total_length):
                c = split_a[total_length - 1 - i]
                new_split.append(c)
                if c[-1] in ['0', '1', '2']:
                    word_length+=1
                if word_length == min_num:
                    break

            new_split.reverse()
            new_splits_a.append(new_split)

        new_splits_b = []
        for split_b in splits_b:
            total_length = len(split_b)
            new_split = []
            word_length = 0
            for i in range(total_length):
                c = split_b[total_length - 1 - i]
                new_split.append(c)
                if c[-1] in ['0', '1', '2']:
                    word_length += 1
                if word_length == min_num:
                    break

            new_split.reverse()
            new_splits_b.append(new_split)

        for split_a in new_splits_a:
            for split_b in new_splits_b:
                if split_a == split_b:
                    return True


        return False

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """
        sentences = text.split('\n')
        last_words = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            if len(words) > 1:
                last_word = words[-1]
                if len(last_word) == 1:
                    last_word = words[-2]
                last_words.append(last_word)

        if len(last_words) != 5:
            return False

        if not(self.rhymes(last_words[0], last_words[1])):
            return False

        if not(self.rhymes(last_words[0], last_words[4])):
            return False

        if not(self.rhymes(last_words[2], last_words[3])):
            return False


        return True

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = input()

    ld = LimerickDetector()
    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))
