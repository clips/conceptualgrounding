"""
This class object is used to control all data during training and testing of the encoders.
"""
import json
from reach import Reach
import numpy as np
from collections import defaultdict
from copy import deepcopy
import fasttext
import random
from tqdm import tqdm
from sklearn.cross_decomposition import CCA

#############################################################
#############################################################
##################      Sampling      #######################
#############################################################
#############################################################


class Sampling:

    def __init__(self, data_infile, fasttext_model_path):

        self.train_embeddings = None
        self.validation_embeddings = None
        self.online_negative_train_embeddings = None
        self.online_negative_validation_embeddings = None
        self.pretrained_prototype_embeddings = None

        with open(data_infile, 'r') as f:
            self.data = json.load(f)

        self.ontology = self.data['ontology']
        self.train_pairs = self.data['train']
        self.validation_pairs = self.data['validation']

        self.train_references = {reference: concept for (concept, reference, positive) in
                                 self.train_pairs}
        self.validation_references = {reference: concept for (concept, reference, positive) in
                                      self.validation_pairs}

        self.exemplar_to_concept = {}
        for concept, exemplars in self.ontology.items():
            for exemplar in exemplars:
                self.exemplar_to_concept[exemplar] = concept

        # load fasttext data
        self.fasttext_model = fasttext.FastText.load_model(fasttext_model_path)

        # create pretrained name embeddings and concept prototypes
        self.vectorize_loaded_data()
        self.load_embeddings()
        self.extract_pretrained_prototype_embeddings()

    def vectorize_string(self, string, norm):

        tokens = string.split()
        token_embeddings = []
        for token in tokens:
            vector = self.fasttext_model.get_word_vector(token)
            if norm:
                vector = Reach.normalize(vector)
            token_embeddings.append(vector)
        token_embeddings = np.array(token_embeddings)

        return token_embeddings

    def create_reach_object(self, strings, normalize=False, outfile=''):

        vectors = []
        for s in tqdm(strings):
            token_embs = self.vectorize_string(s, norm=normalize)
            vector = np.average(np.array(token_embs), axis=0)
            vectors.append(vector)

        reach_object = Reach(vectors, list(strings))

        if outfile:
            reach_object.save_fast_format(outfile)

        return reach_object

    def vectorize_loaded_data(self, outfile=''):

        data = self.data

        all_strings = set()
        for synset in data['complete_ontology'].values():
            all_strings.update(synset)

        for cui, mention, name in data['train']:
            all_strings.add(mention)
        for cui, mention, name in data['validation']:
            all_strings.add(mention)
        for cui, mention, name in data['test']:
            all_strings.add(mention)
        for cui, mention, names in data['zero-shot']:
            all_strings.add(mention)

        reach_object = self.create_reach_object(all_strings, outfile=outfile)

        return reach_object

    def load_embeddings(self, prune=True):

        # possibly prune train embeddings to selected ontology
        self.train_embeddings = deepcopy(self.pretrained_name_embeddings)
        if prune:
            self.train_embeddings.prune(list(self.exemplar_to_concept.keys()))
        print(len(self.train_embeddings.items), len(self.exemplar_to_concept))

        # possibly prune validation embeddings to selected ontology
        self.validation_embeddings = deepcopy(self.pretrained_name_embeddings)
        if prune:
            self.validation_embeddings.prune(list(self.validation_references.keys()))
        print(len(self.validation_embeddings.items), len(self.validation_references))

    def extract_pretrained_prototype_embeddings(self):

        reference_train_embeddings = deepcopy(self.train_embeddings)

        ontology = defaultdict(list, deepcopy(self.ontology))

        # train concept anchors
        for reference, concept in self.train_references.items():
            ontology[concept].append(reference)

        pretrained_prototype_embeddings = {}
        for concept, strings in ontology.items():
            embs = []
            for s in strings:
                emb_index = reference_train_embeddings.items[s]
                emb = reference_train_embeddings.vectors[emb_index]
                embs.append(emb)
            pooled_embedding = np.average(np.array(embs), axis=0)
            pretrained_prototype_embeddings[concept] = pooled_embedding

        self.pretrained_prototype_embeddings = {}
        for reference, concept in self.exemplar_to_concept.items():
            self.pretrained_prototype_embeddings[reference] = pretrained_prototype_embeddings[concept]

        # assign the train prototype embeddings to the relevant validation items
        for validation_item, concept in self.validation_references.items():
            self.pretrained_prototype_embeddings[validation_item] = pretrained_prototype_embeddings[concept]

    def load_online_negative_embeddings(self, embeddings, prune=True):

        # embeddings contain the current representations outputted by the representation network
        # this function enables online negative sampling to help with stochastic training of the network

        self.online_negative_train_embeddings = deepcopy(embeddings)
        self.online_negative_validation_embeddings = deepcopy(embeddings)

        if prune:
            self.online_negative_train_embeddings.prune(list(self.exemplar_to_concept.keys()))
            self.online_negative_validation_embeddings.prune(list(self.validation_references.keys()))

    def positive_sampling(self, validation=False):

        if validation:
            paraphrase_data = self.validation_pairs
        else:
            paraphrase_data = self.train_pairs

        positive_samples = []
        for (concept, anchor, positive) in tqdm(paraphrase_data, disable=True):

            positive_samples.append((concept, anchor, positive))

        return positive_samples

    def negative_name_sampling(self, references, online=True, validation=False, amount_negative=1,
                          threshold=200, verbose=False, random_sampling=False):

        # sampling policy: distance weighted sampling
        # cf. Wu et al. (2017), 'Sampling Matters in Deep Embedding Learning'

        threshold = max(amount_negative, threshold)

        if online:
            train_embeddings = self.online_negative_train_embeddings
            validation_embeddings = self.online_negative_validation_embeddings
        else:
            train_embeddings = self.train_embeddings
            validation_embeddings = self.validation_embeddings

        assert self.online_negative_train_embeddings != None, 'Online negative train embeddings need to be loaded!'
        assert self.online_negative_validation_embeddings != None, 'Online negative validation embeddings need to be loaded!'

        # sample according to proximity
        negative_samples = {}
        for anchor, concept in tqdm(references.items(), total=len(references), disable=not verbose):

            # first sample positive matches to exclude from negative sampling
            positive_exemplars = []
            if concept in self.ontology:
                positive_exemplars = self.ontology[concept]
            positive_exemplar_idxs = [train_embeddings.items[x] for x in positive_exemplars]

            if validation:
                reference_idx = validation_embeddings.items[anchor]
                reference_vector = validation_embeddings.norm_vectors[reference_idx]
            else:
                reference_idx = train_embeddings.items[anchor]
                reference_vector = train_embeddings.norm_vectors[reference_idx]

            if random_sampling:
                negative_paraphrases = set()
                items = list(train_embeddings.items.keys())
                while len(negative_paraphrases) < amount_negative:
                    sampled_negative_paraphrase = random.choice(items)
                    if sampled_negative_paraphrase not in positive_exemplars:
                        negative_paraphrases.add(sampled_negative_paraphrase)
                negative_samples[anchor] = list(negative_paraphrases)
                continue

            # exclude those idxs from the equation, remove
            cosines = train_embeddings.norm_vectors.dot(reference_vector.T)
            top_cosines_idxs = np.argpartition(-cosines, threshold)[:threshold]
            top_cosines = [cosines[i] for i in top_cosines_idxs]
            indexed_cosines = [(i, x) for i, x in zip(top_cosines_idxs, top_cosines) if i not in positive_exemplar_idxs]

            indexes, cosines = zip(*indexed_cosines)
            weights = [1/(np.clip(1-x, 0.000000001, 1)) for x in cosines]
            total_weight = sum(weights)
            probs = [weight/total_weight for weight in weights]

            negative_paraphrases = set()
            while len(negative_paraphrases) < amount_negative:
                sampled_idx = np.random.choice(np.array(indexes), p=probs)
                sampled_negative_paraphrase = train_embeddings.indices[sampled_idx]
                negative_paraphrases.add(sampled_negative_paraphrase)

            negative_samples[anchor] = list(negative_paraphrases)

        return negative_samples

    def fit_cca(self):

        # fits linear CCA constraint and replaces pretrained name embeddings with CCA transformed embeddings

        self.load_embeddings()
        self.extract_pretrained_prototype_embeddings()

        items, vectors = zip(
            *[(k, v) for k, v in self.pretrained_prototype_embeddings.items() if k in self.exemplar_to_concept])
        concept_embs = Reach(vectors, items)

        train_vectors = []
        for x in items:
            train_vectors.append(self.train_embeddings[x])
        train_vectors = Reach.normalize(train_vectors)

        cca = CCA(n_components=self.train_embeddings.size, max_iter=10000)
        cca.fit(train_vectors, concept_embs.norm_vectors)

        # transform all name embeddings using the CCA mapping
        all_name_embeddings = deepcopy(self.pretrained_name_embeddings)
        items = [x for _, x in sorted(all_name_embeddings.indices.items())]
        projected_name_embeddings = cca.transform(all_name_embeddings.norm_vectors)
        new_name_embeddings = Reach(projected_name_embeddings, items)

        self.pretrained_name_embeddings = new_name_embeddings
        self.load_embeddings()
