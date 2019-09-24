from sklearn.datasets import load_files
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm


def load():
    moviedir = "movie_reviews"
    movie_reviews = load_files(moviedir)
    return movie_reviews


def word_counts_each_movie(movie_reviews):
    """
    Given movie reviews, suppose 2000 docs, Vacabulary is V, then it returns a list of dictionary:
    [doc1: {word1: count, word2: count, ... wordV: count},
    ...
    doc2000: {}
    ]
    :param movie_reviews:
    :return: word_counts
    """
    vector_setting = CountVectorizer(min_df=1, tokenizer=nltk.word_tokenize)
    word_counts = vector_setting.fit_transform(movie_reviews.data)
    return word_counts


def word_counts_to_tfidf(word_counts):
    """
    Consider tf-idf, redo the word_counts to tf-idf feature
    :param word_counts:
    :return:
    """
    tfidf_transformer = TfidfTransformer()
    word_tfidf = tfidf_transformer.fit_transform(word_counts)
    return word_tfidf


def split_train_test(movie_reviews, feature):
    """
    split word_counts or word_tfidf to train and test set
    :param movie_reviews: use it as label
    :param feature: either word_counts or word_tfidf
    :return: splited set
    """
    feature_train, feature_test, label_train, label_test = train_test_split(
        feature, movie_reviews.target, test_size=0.20, random_state=12)
    return feature_train, feature_test, label_train, label_test


def cm_to_metrics(cm):
    """

    :param cm: confusion matrix
    :return: A dictionary contains precision, recall, F1, and accuracy
    """
    metrics = {}
    tp, fn, fp, tn = cm.ravel()
    metrics['precision'] = tp/(tp+fp)
    metrics['recall'] = tp/(tp+fn)
    metrics['accuracy'] = (tp+tn)/(tp+fp+tn+fn)
    metrics['F1'] = (2*metrics['precision']*metrics['recall']) / (metrics['precision'] + metrics['recall'])
    return metrics


reviews = load()

############################################################
# This part use bag of word feature for NB
############################################################
print("*" * 20)
print("Bag of Word for NB\n")
counts_word = word_counts_each_movie(reviews)
train, test, train_label, test_label = split_train_test(reviews, counts_word)

clf = MultinomialNB().fit(train, train_label)
y_pred = clf.predict(test)

cm = confusion_matrix(test_label, y_pred)
metrics = cm_to_metrics(cm)
print("Precision : {:.5f}%\n Recall : {:.5f}%\n F1 : {:.5f}%\n Accuracy : {:.5f}%\n\n"
      .format(metrics['precision']*100, metrics['recall']*100, metrics['F1']*100, metrics['accuracy']*100))


###########################################################
# Bag of Word for SVM
###########################################################
print("*" * 20)
print("Bag of Word for SVM\n")
clf = svm.SVC(gamma='scale')
clf.fit(train, train_label)
y_pred = clf.predict(test)
cm = confusion_matrix(test_label, y_pred)
metrics = cm_to_metrics(cm)
print("Precision : {:.5f}%\n Recall : {:.5f}%\n F1 : {:.5f}%\n Accuracy : {:.5f}%\n\n"
      .format(metrics['precision']*100, metrics['recall']*100, metrics['F1']*100, metrics['accuracy']*100))



############################################################
# This part use tf-idf feature for SVM
############################################################
print("*" * 20)
print("TF-IDF for SVM\n")
clf = svm.SVC(gamma='scale')
word_tfidf = word_counts_to_tfidf(counts_word)
train, test, train_label, test_label = split_train_test(reviews, word_tfidf)
clf.fit(train, train_label)
y_pred = clf.predict(test)
cm = confusion_matrix(test_label, y_pred)
metrics = cm_to_metrics(cm)
print("Precision : {:.5f}%\n Recall : {:.5f}%\n F1 : {:.5f}%\n Accuracy : {:.5f}%\n\n"
      .format(metrics['precision']*100, metrics['recall']*100, metrics['F1']*100, metrics['accuracy']*100))

