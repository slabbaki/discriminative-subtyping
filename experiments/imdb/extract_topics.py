from __future__ import division, print_function

import os
import sys
import argparse
import numpy as np
from scipy import sparse as sp

# from sklearn.decomposition import LatentDirichletAllocation as LDA
from lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score

def load_data(data_path, vocab_size=40000, verbose=1):
    if verbose:
        sys.stdout.write("Loading data..."); sys.stdout.flush()

    with open(os.path.join(data_path, 'raw', 'imdb.vocab')) as fp:
        vocab = [word.strip() for word in fp.readlines()]
    X_train, y_train = load_svmlight_file(
        os.path.join(data_path, 'raw/train/labeledBow.feat'),
        n_features=len(vocab))
    X_train_unsup, _ = load_svmlight_file(
        os.path.join(data_path, 'raw/train/unsupBow.feat'),
        n_features=len(vocab))
    X_test, y_test = load_svmlight_file(
        os.path.join(data_path, 'raw/test/labeledBow.feat'),
        n_features=len(vocab))

    # Pre-process the data (reduce the vocabulary size)
    word_ids = np.nonzero(
        [w not in ENGLISH_STOP_WORDS for w in vocab])[0][:vocab_size]
    X_train = X_train[:, word_ids]
    X_train_unsup = X_train_unsup[:, word_ids]
    X_test = X_test[:, word_ids]

    # Pre-process labels
    y_train = (y_train > 5).astype('int')
    y_test = (y_test > 5).astype('int')

    if verbose:
        sys.stdout.write("Done.\n")

    return vocab, word_ids, \
           (X_train, y_train), (X_test, y_test), X_train_unsup


def compute_topics(X_train, X_test, n_topics,
                   learning_method='batch', max_iter=1000, batch_size=128,
                   n_jobs=-1, eval_every=10, verbose=1, seed=42):
    lda = LDA(n_topics=n_topics,
              #learning_method=learning_method,
              #max_iter=max_iter,
              alpha=0.1,
              eta=0.01,
              n_iter=max_iter,
              #batch_size=batch_size,
              #evaluate_every=eval_every,
              #n_jobs=n_jobs,
              #verbose=verbose,
              random_state=np.random.RandomState(seed))

    if verbose:
        sys.stdout.write("Fitting a topic model data...\n"); sys.stdout.flush()

    lda.fit(X_train.astype(np.int64))
    topic_word = lda.components_
    T_train = lda.transform(X_train.astype(np.int64))
    T_test = lda.transform(X_test.astype(np.int64))

    # train_perplexity = lda.perplexity(X_train)
    # test_perplexity = lda.perplexity(X_test)

    if verbose:
        # sys.stdout.write("Train perplexity: %f\n" % train_perplexity)
        # sys.stdout.write("Test perplexity: %f\n" % test_perplexity)
        sys.stdout.write("Train loglik: %f\n" % lda.loglikelihood())
        sys.stdout.write("Done.\n")

    return T_train, T_test, topic_word


def save_topics(data_path, T_train, T_test, topic_word,
                n_topics, word_ids, vocab_size, use_unlabeled):
    prefix = 'utrain_' if use_unlabeled else 'train_'
    path = os.path.join(
        data_path, 'topics', prefix + '%d_%d.npy' % (n_topics, vocab_size))
    np.save(path, T_train)
    prefix = 'utest_' if use_unlabeled else 'test_'
    path = os.path.join(
        data_path, 'topics', prefix + '%d_%d.npy' % (n_topics, vocab_size))
    np.save(path, T_test)
    word_ids_path = os.path.join(
        data_path, 'topics', 'word_ids_%d.npy' % vocab_size)
    np.save(word_ids_path, word_ids)
    prefix = 'utopic_word_' if use_unlabeled else 'topic_word_'
    path = os.path.join(
        data_path, 'topics', prefix + '%d_%d.npy' % (n_topics, vocab_size))
    np.save(path, topic_word)


def main(args):
    data_path = os.path.join(os.environ['DATA_PATH'], 'IMDB')
    vocab, word_ids, (X_train, y_train), (X_test, y_test), X_train_unsup = \
        load_data(data_path, vocab_size=args.vocab_size)

    if args.use_unlabeled:
        X_train = sp.vstack([X_train, X_train_unsup])

    # Fit a topic model
    T_train, T_test, topic_word = \
        compute_topics(X_train, X_test, args.n_topics,
                       max_iter=args.max_iter, n_jobs=args.n_jobs)
    save_topics(data_path, T_train, T_test, topic_word,
                args.n_topics, word_ids, args.vocab_size, args.use_unlabeled)

    if args.use_unlabeled:
        T_train = T_train[:len(y_train), :]

    # Check the predictiveness of the topic representation
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(T_train, y_train)
    y_pred = lr.predict(T_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_topics", type=int,
                        help="number of topics")
    parser.add_argument("--max_iter", type=int, default=30,
                        help="max number of LDA iterations")
    parser.add_argument("--vocab_size", type=int, default=40000,
                        help="vocabulary size")
    parser.add_argument("--use_unlabeled", action="store_true",
                        help="whether to use unlabeled part of the data")
    parser.add_argument("-n", "--n_jobs", type=int, default=-1,
                        help="number of parallel jobs to run")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
