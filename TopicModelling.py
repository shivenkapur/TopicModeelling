# %%
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import string
import gensim
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def getTopics(document):

    stopwords_ = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean_document(document):
        stopwordremoval = " ".join(
            [i for i in document.lower().split() if i not in stopwords_])
        punctuationremoval = ''.join(
            ch for ch in stopwordremoval if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word)
                              for word in punctuationremoval.split())
        return normalized

    final_doc = clean(document).split()
    dictionary = corpora.Dictionary([final_doc])
    DT_matrix = [dictionary.doc2bow(final_doc)]

    Lda_object = gensim.models.ldamodel.LdaModel
    lda_model_1 = Lda_object(DT_matrix, num_topics=2, id2word=dictionary)
    return lda_model_1.print_topics(num_topics=2, num_words=5)
