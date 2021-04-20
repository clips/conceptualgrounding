import json
import numpy as np
from tqdm import tqdm
from reach import Reach

###############################################################
###############################################################
##################      RANKING      ##########################
###############################################################
###############################################################


class SynonymRetrieval:

    def __init__(self):

        self.ontology = None
        self.exemplar_to_concept = None

        self.train_vectors = None
        self.test_vectors = None

        self.verbose = False

    def load_ontology(self, ontology):

        self.ontology = ontology

        self.exemplar_to_concept = {}
        for concept, exemplars in self.ontology.items():
            for exemplar in exemplars:
                self.exemplar_to_concept[exemplar] = concept

    def load_train_vectors(self, embeddings_infile, prune=True):
        # load vectors
        print('Loading vectors...')
        self.train_vectors = Reach.load_fast_format(embeddings_infile)
        if prune:
            # prune embeddings to selected target ontology
            assert self.exemplar_to_concept
            self.train_vectors.prune(list(self.exemplar_to_concept.keys()))
        print(len(self.train_vectors.items), len(self.exemplar_to_concept))

    def load_train_vectors_object(self, embedding_object, prune=True):
        # load vectors
        print('Loading vectors...')
        self.train_vectors = embedding_object
        if prune:
            # prune embeddings to selected target ontology
            assert self.exemplar_to_concept
            self.train_vectors.prune(list(self.exemplar_to_concept.keys()))
        print(len(self.train_vectors.items), len(self.exemplar_to_concept))

    def load_test_vectors(self, embeddings_infile):
        # load vectors
        print('Loading vectors...')
        self.test_vectors = Reach.load_fast_format(embeddings_infile)

    def load_test_vectors_object(self, embedding_object):
        # load vectors
        print('Loading vectors...')
        self.test_vectors = embedding_object

    def synonym_retrieval_train(self, train_pairs, outfile=''):

        assert self.train_vectors != None, 'No vectors are loaded yet!'

        complete_ranking = []
        for instance in tqdm(train_pairs.items(), disable=False):

            reference, concept = instance

            synonyms = self.ontology[concept]
            without_reference = [x for x in synonyms if x != reference]
            if without_reference:
                synonyms = without_reference

            synonym_idxs = [self.train_vectors.items[syn] for syn in synonyms]

            reference_idx = self.train_vectors.items[reference]

            # calculate distances
            reference_vector = self.train_vectors.norm_vectors[reference_idx]
            scores = self.train_vectors.norm_vectors.dot(reference_vector.T)

            # extract ranking
            mask = [1 if x == reference_idx else 0 for x in range(len(self.train_vectors.items))]
            scores = np.ma.array(scores, mask=mask)
            ranking = np.argsort(-scores)
            ranks = [np.where(ranking == synonym_idx)[0][0] for synonym_idx in synonym_idxs]
            assert ranks
            ranks, synonyms = zip(*sorted(zip(ranks, synonyms)))
            instance = (concept, reference, synonyms)
            complete_ranking.append((instance, ranks))

        if outfile:
            print('Saving...')
            with open(outfile, 'w') as f:
                json.dump(complete_ranking, f)

        instances, rankings = zip(*complete_ranking)
        print(round(self.mean_average_precision(rankings), 2), '&',
              round(self.ranking_accuracy(rankings), 2), '&',
              round(self.mean_reciprocal_rank(rankings), 2), '&')

        return complete_ranking

    def synonym_retrieval_test(self, test_pairs, outfile=''):

        assert self.train_vectors != None, 'No train vectors are loaded yet!'
        assert self.test_vectors != None, 'No test vectors are loaded yet!'

        complete_ranking = []
        for instance in tqdm(test_pairs.items(), disable=False):

            reference, concept = instance

            synonyms = self.ontology[concept]
            synonym_idxs = [self.train_vectors.items[syn] for syn in synonyms]

            reference_idx = self.test_vectors.items[reference]

            # calculate distances
            reference_vector = self.test_vectors.norm_vectors[reference_idx]
            scores = self.train_vectors.norm_vectors.dot(reference_vector.T)

            # extract ranking
            ranking = np.argsort(-scores)
            ranks = [np.where(ranking == synonym_idx)[0][0] for synonym_idx in synonym_idxs]
            assert ranks
            ranks, synonyms = zip(*sorted(zip(ranks, synonyms)))
            instance = (concept, reference, synonyms)
            complete_ranking.append((instance, ranks))

        if outfile:
            print('Saving...')
            with open(outfile, 'w') as f:
                json.dump(complete_ranking, f)

        instances, rankings = zip(*complete_ranking)
        print(round(self.mean_average_precision(rankings), 2), '&',
              round(self.ranking_accuracy(rankings), 2), '&',
              round(self.mean_reciprocal_rank(rankings), 2), '&')

        return complete_ranking

    def synonym_retrieval_zeroshot(self, zeroshot_pairs, isolated=False, outfile=''):

        assert self.train_vectors != None, 'No train vectors are loaded yet!'
        assert self.test_vectors != None, 'No test vectors are loaded yet!'

        # new setting: add ALL zeroshot data to train data to cause more confusion
        train_items = [x for _, x in sorted(self.train_vectors.indices.items())]
        train_vectors = self.train_vectors.vectors

        zeroshot_items = set()
        for concept, reference, synonyms in zeroshot_pairs:
            zeroshot_items.add(reference)
            zeroshot_items.update(synonyms)
        zeroshot_items = sorted(zeroshot_items)
        zeroshot_vectors = []
        for zeroshot_item in zeroshot_items:
            zeroshot_vectors.append(self.test_vectors[zeroshot_item])
        if isolated:
            fused_vectors = Reach(zeroshot_vectors, zeroshot_items)
        else:
            all_items = train_items + zeroshot_items
            zeroshot_vectors = np.array(zeroshot_vectors)
            all_vectors = np.concatenate((train_vectors, zeroshot_vectors), axis=0)
            fused_vectors = Reach(all_vectors, all_items)

        # now rank
        complete_ranking = []
        for instance in tqdm(zeroshot_pairs, disable=False):

            concept, reference, synonyms = instance

            synonym_idxs = [fused_vectors.items[syn] for syn in synonyms]

            reference_idx = fused_vectors.items[reference]

            # calculate distances
            reference_vector = fused_vectors.norm_vectors[reference_idx]
            scores = fused_vectors.norm_vectors.dot(reference_vector.T)

            # extract ranking
            mask = [1 if x == reference_idx else 0 for x in range(len(fused_vectors.items))]
            scores = np.ma.array(scores, mask=mask)
            ranking = np.argsort(-scores)
            ranks = [np.where(ranking == synonym_idx)[0][0] for synonym_idx in synonym_idxs]
            assert ranks
            ranks, synonyms = zip(*sorted(zip(ranks, synonyms)))
            instance = (concept, reference, synonyms)
            complete_ranking.append((instance, ranks))

        if outfile:
            print('Saving...')
            with open(outfile, 'w') as f:
                json.dump(complete_ranking, f)

        instances, rankings = zip(*complete_ranking)
        print(round(self.mean_average_precision(rankings), 2), '&',
              round(self.ranking_accuracy(rankings), 2), '&',
              round(self.mean_reciprocal_rank(rankings), 2), '&')

        return complete_ranking

    @staticmethod
    def precision_at_k(r, k):
        assert k >= 1
        r = np.asarray(r)[:k] != 0
        if r.size != k:
            raise ValueError('Relevance score length < k')
        return np.mean(r)

    def average_precision(self, r):
        r = np.asarray(r) != 0
        out = [self.precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
        if not out:
            return 0.
        return np.mean(out)

    @staticmethod
    def convert_ranks(ranks):
        r = np.zeros(max(ranks) + 1)
        for rank in ranks:
            r[rank] = 1

        return r

    def mean_average_precision(self, ranking):
        avg_precs = []
        for ranks in tqdm(ranking, disable=not self.verbose):
            # convert ranks to binary labels
            r = self.convert_ranks(ranks)
            avg_prec = self.average_precision(r)
            avg_precs.append(avg_prec)

        mAP = np.mean(avg_precs)

        return mAP

    @staticmethod
    def mean_reciprocal_rank(ranking):
        reciprocal_ranks = []
        for ranks in ranking:
            reciprocal_rank = 1 / (ranks[0] + 1)
            reciprocal_ranks.append(reciprocal_rank)

        mrr = np.mean(reciprocal_ranks)

        return mrr

    @staticmethod
    def ranking_accuracy(ranking):
        corrects = 0
        for ranks in ranking:
            if ranks[0] == 0:
                corrects += 1

        accuracy = corrects / len(ranking)

        return accuracy
