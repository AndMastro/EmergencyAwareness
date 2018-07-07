import string

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer

stemmer = EnglishStemmer()

def stemming_tokenizer(text):
	stemmed_text = [stemmer.stem(word) for word in word_tokenize(text, language='english')]
	return stemmed_text

def tweet_tokenizer(text):
	stemmed_text = [stemmer.stem(word) for word in TweetTokenizer(preserve_case = False, strip_handles=True, reduce_len=True).tokenize(text)]
	return stemmed_text

def std_tokenizer(text):
	stemmed_text = [word for word in word_tokenize(text, language='english')]
	return stemmed_text