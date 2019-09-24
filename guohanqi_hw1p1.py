import os
import os.path as osp
import numpy as np
import json
import sys


class NaiveBayes(object):
    def __init__(self, folds):
        self.folds = folds
        self.V = []
        self.counts = {}  # counts for word,  a dictionary of dictionary, outer key is class, inner key is word,
        # value is the number of time the word occurrence. {"pos":{"hello": appear times, ..., ""}, "neg":{,,,}}
        self.classes = ["pos", "neg"]
        self.train = {}  # training set, {"pos": [list of files], "neg": [list of files]}
        self.test = {}  # test set, same format as training set
        self.path = {}  # path set, {"pos": path-to-pos-dir, "neg": path-to-neg-dir}
        self.magadoc = {}  # magadoc of docs, {"pos": [list of readlines of each doc], "neg": []}
        self.total_magadoc = []  # magadoc of total docs, [list of readlines of all docs]
        self.logpossibility = {}
        self.logprior = {}

    def fold_selector(self):
        fold_range = []
        for fold in self.folds:
            index = int(fold[4:])
            index = 100 * index
            fold_range.append(index)
        return fold_range

    def construct_train_fold(self):
        """
        This function construct train folds in the corpus.
        :return: self.train as a dictionary
        """
        fold_range = self.fold_selector()
        for cls in self.classes:
            path = osp.join('movie_reviews', cls)
            self.train[cls] = []
            for docs in os.listdir(path):
                if int(docs[2:5]) in range(fold_range[0] - 100, fold_range[0]) or int(docs[2:5]) in range(
                        fold_range[1] - 100, fold_range[1]):  # should be 200
                    self.train[cls].append(docs)
            self.path[cls] = path

    def construct_test_fold(self):
        """
        This function construct test folds in the corpus.
        :return: self.test as a dictionary
        """
        fold_range = self.fold_selector()
        for cls in self.classes:
            path = osp.join('movie_reviews', cls)
            self.test[cls] = []
            for docs in os.listdir(path):
                if int(docs[2:5]) in range(fold_range[0] - 100, fold_range[0]):
                    self.test[cls].append(docs)
            self.path[cls] = path

    def construct_fold(self):
        """
        This function construct test and train fold in the corpus.
        :return:
        """
        fold_range = self.fold_selector()
        for cls in self.classes:
            path = osp.join('movie_reviews', cls)
            self.train[cls] = []
            self.test[cls] = []
            for docs in os.listdir(path):
                if int(docs[2:5]) in range(fold_range[0] - 100, fold_range[0]) or int(docs[2:5]) in range(
                        fold_range[1] - 100, fold_range[1]):  # should be 200
                    self.train[cls].append(docs)
                elif int(docs[2:5]) < 300:  # should be 300
                    self.test[cls].append(docs)
            self.path[cls] = path

    def construct_maga_file(self):
        """
        This Function is working to construct maga_file for positive label, negative label and both.
        :return: negative magadoc, positive magadoc, total magadoc
        """
        for cls in self.classes:
            self.magadoc[cls] = []
            for file in self.train[cls]:
                file_path = osp.join(self.path[cls], file)
                with open(file_path) as f:
                    lines = f.readlines()
                    self.magadoc[cls].extend(lines)
            self.total_magadoc.extend(self.magadoc[cls])

    def wordCount(self):
        """
        This function counts words in class
        :return: a dictionary of dictionary, outer key is class, inner key is word, value is the
         number of time the word occurrence.
        """
        self.V = set(word.lower() for passage in self.total_magadoc for word in passage.split(" "))

        # Laplace Smoothing
        for cls in self.classes:
            self.counts[cls] = {}
            for word in self.V:
                self.counts[cls][word] = 1

        for cls in self.classes:
            for doc in self.magadoc[cls]:
                words = doc.split(" ")
                for word in words:
                    self.counts[cls][word] += 1

    def trainMyBayes(self):
        self.construct_train_fold()
        self.construct_maga_file()
        self.wordCount()
        N = len(self.total_magadoc)

        for cls in self.classes:
            N_cls = len(self.train[cls])

            # Calculate P(C)
            print('*'*20)
            print("Calculating P(C) at {}\n".format(cls))
            self.logprior[cls] = np.log(N_cls / N)
            print('*' * 20)
            print("P(C) calculate complete!\n")
            self.logpossibility[cls] = {}
            # sum(w-V) count(w, cj)
            print('*' * 20)
            print("Calculating P(w|C) at {}\n".format(cls))
            total_count = 0
            for word in self.V:
                total_count += self.counts[cls][word]

            for word in self.V:
                count = self.counts[cls][word]
                # Calculate P(w_i, c_j)
                self.logpossibility[cls][word] = np.log(count / (total_count + len(self.V)))
            print('*' * 20)
            print("P(w|C) calculate complete\n")
        result = {'logpossibility': self.logpossibility, 'logprior': self.logprior, 'V': list(self.V),
                  'classes': self.classes}
        with open('result.json', 'w') as f:
            json.dump(result, f)
        print('*' * 20)
        print("Parameter saved to result.json")

    def test_target(self, target):
        predict = {}
        test_words = []
        with open(target) as f:
            doc = f.readlines()
        for sentence in doc:
            test_words.extend(sentence.split(" "))
        for cls in self.classes:
            predict[cls] = self.logprior[cls]
            for word in test_words:
                if word in self.V:
                    predict[cls] += self.logpossibility[cls][word]
        return predict

    def test_fold(self):
        print('*'*30)
        print("Construct Test Folds\n")
        self.construct_test_fold()
        print('*' * 30)
        print("Load Parameters\n")
        with open('result.json') as f:
            result = json.load(f)
        self.V = result['V']
        self.classes = result['classes']
        self.logprior = result['logprior']
        self.logpossibility = result['logpossibility']

        correct = 0
        total_count = 0
        true = {}
        false = {}
        for cls in self.classes:
            docs = self.test[cls]
            false[cls] = 0
            true[cls] = 0
            for doc in docs:
                total_count += 1
                path = osp.join(self.path[cls], doc)
                predict = self.test_target(path)
                v = list(predict.values())  # find the greater value in predict dictionary.
                V = v.index(max(v))
                k = list(predict.keys())
                result = k[V]  # result is either "pos" or "neg"
                if result == cls:
                    correct += 1
                    true[cls] += 1
                else:
                    false[cls] += 1
                print("#{}/200   current correct predict is {}/{}".format(total_count, correct, total_count))
                print("#{}/200   current accuracy {:12.4f}% \n".format(total_count, correct * 100 / total_count))

        Recall = true["pos"] / (true["pos"] + false["pos"])
        Precision = true["pos"] / (true["pos"] + false["neg"])
        acc = (true["pos"] + true["neg"]) / (true["pos"] + true["neg"] + false["pos"] + false["neg"])
        f1 = 2 * Precision * Recall / (Precision + Recall)
        print("*" * 20)
        print("#####    Recall = {:.4f}".format(Recall))
        print("#####    Precision = {:.4f}".format(Precision))
        print("#####    Accuracy = {:.4f}".format(acc))
        print("#####    F1 = {:.4f}".format(f1))


if __name__ == '__main__':
    command = sys.argv[1:]
    if command[0] == 'train':
        folds = command[1:]
        NaiveBayes = NaiveBayes(folds)
        NaiveBayes.trainMyBayes()

    elif command[0] == 'test':
        test_fold = command[1:]
        NaiveBayes = NaiveBayes(test_fold)
        NaiveBayes.test_fold()