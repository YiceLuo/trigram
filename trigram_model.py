import sys
from collections import defaultdict
import math
import random
import os
import os.path


def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):

    if n==1:
        sequence=['START']+sequence
    sequence.append('STOP')
    return [tuple([sequence[i+j] if i+j>=0 else 'START' for j in range(-n+1,1)]) for i in range(len(sequence))]


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.total_uni=sum([self.unigramcounts[i] for i in self.unigramcounts])


    def count_ngrams(self, corpus):
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)
        for sentence in corpus:
            for ngram in get_ngrams(sentence,1):
                self.unigramcounts[ngram]+=1
            for gram in get_ngrams(sentence,2): 
                self.bigramcounts[gram]+=1
            for gram in get_ngrams(sentence,3): 
                self.trigramcounts[gram]+=1

        return

    def raw_trigram_probability(self,trigram):

        if trigram[:-1]==('START','START'):
            deno=self.unigramcounts[('START',)]
        else:
            deno=self.bigramcounts[trigram[:-1]]
        if deno==0:
            return 0
        return self.trigramcounts[trigram]/deno


    def raw_bigram_probability(self, bigram):

        deno=self.unigramcounts[bigram[:-1]]
        if deno == 0:
            return 0
        return self.bigramcounts[bigram]/deno
    
    def raw_unigram_probability(self, unigram):

        return self.unigramcounts[unigram]/self.total_uni

    def generate_sentence(self,t=20): 
        sentence=["START","START"]
        for i in range(t):
            tris=[j for j in self.trigramcounts if j[0]==sentence[-2] and j[1]==sentence[-1]]
            weights=[self.trigramcounts[j] for j in tris]
            words=[j[2] for j in tris]
            word=random.choices(words,weights,k=1)
            if word==['UNK']:
                continue
            if word==['STOP']:
                return sentence[2:]
            sentence+=word
        result=sentence[2:-1]
        return result           

    def smoothed_trigram_probability(self, trigram):

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        smprob=lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:])+lambda3*self.raw_unigram_probability(trigram[2:])
        
        return smprob
        
    def sentence_logprob(self, sentence):

        tris=get_ngrams(sentence,3)
        prob=sum([math.log2(self.smoothed_trigram_probability(i)) if self.smoothed_trigram_probability(i)!=0 else float("-Inf") for i in tris])
        return prob

    def perplexity(self, corpus):

        sens=[i for i in corpus]
        M=sum([len(i) for i in sens])
        l=[self.sentence_logprob(i) for i in sens]
        a=sum(l)/M
        return 2**(-a) 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0   
        hpp=[]
        lpp=[]
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            hpp.append(pp) 
            pp = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            lpp.append(pp)
        total=total+len(hpp)
        correct=correct+sum([hpp[i]<lpp[i] for i in range(len(hpp))])
        hpp=[]
        lpp=[]        
        for f in os.listdir(testdir2):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            hpp.append(pp) 
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            lpp.append(pp) 
        total=total+len(hpp)
        correct=correct+sum([hpp[i]>lpp[i] for i in range(len(hpp))])
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # cd ./Documents/cu/nlp/4705
    # or run the script from the command line with 
    # $ python -i trigram_model.py ./hw1_data/brown_train.txt ./hw1_data/brown_test.txt
    # >>> 
    
    # Testing perplexity: 
    #dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    #pp = model.perplexity(dev_corpus)
    print("perplexity")
    print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt', './hw1_data/ets_toefl_data/train_low.txt', "./hw1_data/ets_toefl_data/test_high", "./hw1_data/ets_toefl_data/test_low")
    print("acc")
    print(acc)

