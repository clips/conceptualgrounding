import json
import random
from collections import defaultdict
from copy import deepcopy
import time

import numpy as np
import torch
from tqdm import tqdm

from synonym_retrieval import SynonymRetrieval
from encoder_base import BaseLSTM


######################################################
######################################################
##################      LSTM BNE      ################
######################################################
######################################################

class EncoderBNE(BaseLSTM):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.sampling.extract_pretrained_prototype_embeddings()

    def batch_step(self, positive_samples_batch, negative_samples, normalize=True, train=True):

        losses = {}

        if train:
            # set model back to train mode
            self.model.train()
            # clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

        ################################################
        #####          SYNONYM SIMILARITY       ########
        ################################################
        anchor_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            anchor_name_token_vectors = self.sampling.vectorize_string(anchor, norm=normalize)
            anchor_name_batch.append(torch.FloatTensor(anchor_name_token_vectors).to(self.device))
        online_anchor_name_batch = self.forward_lstm(anchor_name_batch)

        if train:
            assert self.model.training

        positive_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            positive_name_token_vectors = self.sampling.vectorize_string(positive, norm=normalize)
            positive_name_batch.append(torch.FloatTensor(positive_name_token_vectors).to(self.device))
        online_positive_name_batch = self.forward_lstm(positive_name_batch)

        positive_name_distance = self.positive_distance(online_anchor_name_batch, online_positive_name_batch)
        losses['positive_name_distance'] = positive_name_distance

        if train:
            assert self.model.training

        # first: negative sampling
        negative_name_samples = negative_samples['name']

        # then further processing
        negative_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            negative_name_vectors = []
            for negative in negative_name_samples[anchor]:
                negative_name_token_vectors = self.sampling.vectorize_string(negative, norm=normalize)
                negative_name_vectors.append(torch.FloatTensor(negative_name_token_vectors).to(self.device))
            online_negative_name_batch = self.forward_lstm(negative_name_vectors)
            negative_name_batch.append(online_negative_name_batch)
        online_negative_name_batch = torch.stack(negative_name_batch)

        negative_name_distance = self.negative_distance(online_anchor_name_batch, online_negative_name_batch)
        triplet_name_vs_name = self.triplet_loss(positive_name_distance, negative_name_distance)
        losses['negative_name_distance'] = negative_name_distance
        losses['synonym'] = triplet_name_vs_name

        ##################################################
        #####     CONCEPTUAL MEANINGFULNESS       ########
        ##################################################

        if train:
            assert self.model.training

        pretrained_prototype_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            pretrained_prototype = self.sampling.pretrained_prototype_embeddings[anchor]
            pretrained_prototype_batch.append(torch.FloatTensor(pretrained_prototype).to(self.device))
        pretrained_prototype_batch = torch.stack(pretrained_prototype_batch)

        online_name_to_pretrained_prototype = self.pretrained_loss(online_anchor_name_batch,
                                                                       pretrained_prototype_batch)
        losses['conceptual'] = online_name_to_pretrained_prototype

        ##################################################
        #####     CONTEXTUAL MEANINGFULNESS       ########
        ##################################################

        if train:
            assert self.model.training

        pretrained_name_batch = []
        for (concept, anchor, positive) in positive_samples_batch:
            if train:
                anchor_embeddings = self.sampling.train_embeddings
            else:
                anchor_embeddings = self.sampling.validation_embeddings
            anchor_name_idx = anchor_embeddings.items[anchor]
            if normalize:
                anchor_vector = anchor_embeddings.norm_vectors[anchor_name_idx]
            else:
                anchor_vector = anchor_embeddings.vectors[anchor_name_idx]
            pretrained_name_batch.append(torch.FloatTensor(anchor_vector).to(self.device))
        pretrained_name_batch = torch.stack(pretrained_name_batch)

        online_name_to_pretrained_name = self.pretrained_loss(online_anchor_name_batch, pretrained_name_batch)
        losses['contextual'] = online_name_to_pretrained_name

        ################################################
        #####          TOTAL LOSS               ########
        ################################################
        loss = self.combined_loss(losses)
        losses['loss'] = loss

        if train:
            # getting gradients w.r.t. parameters
            loss.backward()
            # updating parameters
            self.optimizer.step()

        losses = {k: v.item() for k, v in losses.items()}

        return losses

    @staticmethod
    def process_losses(losses):

        avg_losses = defaultdict(list)
        for loss_dict in losses:
            for loss_type, loss in loss_dict.items():
                avg_losses[loss_type].append(loss)
        avg_losses = {k: np.mean(v) for k, v in avg_losses.items()}

        return avg_losses

    def train(self, include_validation=True, stopping_criterion=True, amount_negative_names=1, reinitialize=False,
              normalize=True, verbose=True, update_iteration=50, outfile=''):

        if reinitialize:
            self.reinitialize_model()
            self.loss_cache = defaultdict(dict)
            self.stopping_criterion_cache = {}

        self.amount_negative_names = amount_negative_names
        assert self.amount_negative_names

        positive_train_samples = self.sampling.positive_sampling(validation=False)
        positive_validation_samples = self.sampling.positive_sampling(validation=True)

        stopping_criterion_cache = {}
        if stopping_criterion:
            self.num_epochs = 1000

        torch.requires_grad = True
        # iterate over epochs
        start = time.time()
        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, disable=True):

            # determine epoch ref
            if reinitialize:
                epoch_ref = epoch
            else:
                epoch_ref = max(self.loss_cache) + 1 if self.loss_cache else 1
            print('Started epoch {}'.format(epoch_ref))

            print('Train negative sampling...')
            embeddings = self.extract_online_lstm_embeddings(prune=True, normalize=normalize)
            self.sampling.load_online_negative_embeddings(embeddings, prune=True)
            self.sampling.extract_online_prototype_embeddings()

            negative_train_samples = {}
            references = {anchor: concept for (concept, anchor, positive) in positive_train_samples}
            negative_name_samples = self.sampling.negative_name_sampling(references, online=True, validation=False,
                                                                         amount_negative=self.amount_negative_names,
                                                                         verbose=True)
            negative_train_samples['name'] = negative_name_samples

            # iterate over shuffled batches
            print('Training...')
            train_losses = []
            iteration = 0
            random.shuffle(positive_train_samples)
            for i in tqdm(range(0, len(positive_train_samples), self.batch_size), disable=not verbose):
                batch = positive_train_samples[i: i + self.batch_size]
                train_loss = self.batch_step(batch, negative_samples=negative_train_samples, normalize=normalize, train=True)
                train_losses.append(train_loss)
                iteration += 1
                if verbose:
                    if iteration % update_iteration == 0:
                        avg_train_losses = self.process_losses(train_losses)
                        print('Iteration: {}. Average training losses: {}'.format(iteration, avg_train_losses))

            # update training and/or validation loss
            avg_train_losses = self.process_losses(train_losses)
            self.loss_cache[epoch_ref]['train'] = avg_train_losses
            print('Epoch: {}. Average training losses: {}.'.format(epoch_ref,  avg_train_losses))

            # optionally calculate validation loss
            if include_validation:

                print('Validation negative sampling...')
                embeddings = self.extract_online_lstm_embeddings(prune=True, normalize=normalize)
                self.sampling.load_online_negative_embeddings(embeddings, prune=True)
                self.sampling.extract_online_prototype_embeddings()

                negative_validation_samples = {}
                references = {anchor: concept for (concept, anchor, positive) in positive_validation_samples}
                negative_name_samples = self.sampling.negative_name_sampling(references, online=True, validation=True,
                                                                             amount_negative=self.amount_negative_names,
                                                                             verbose=True)
                negative_validation_samples['name'] = negative_name_samples

                print('Validating...')
                validation_losses = []
                random.shuffle(positive_validation_samples)
                for i in tqdm(range(0, len(positive_validation_samples), self.batch_size), disable=not verbose):
                    batch = positive_validation_samples[i: i + self.batch_size]
                    validation_loss = self.batch_step(batch, negative_samples=negative_validation_samples,
                                                           normalize=normalize, train=False)
                    validation_losses.append(validation_loss)

                avg_validation_losses = self.process_losses(validation_losses)
                self.loss_cache[epoch_ref]['validation'] = avg_validation_losses
                print('Epoch: {}. Validation losses: {}.'.format(epoch_ref, avg_validation_losses))

            # optionally calculate stopping criterion
            print('Calculating validation MRR as stopping criterion...')
            if stopping_criterion:
                validation_mrr = self.stopping_criterion()
                stopping_criterion_cache[epoch_ref] = validation_mrr
                try:
                    previous_mrr = stopping_criterion_cache[epoch_ref - 1]
                    if validation_mrr < previous_mrr:
                        data = {'losses': self.loss_cache,
                                'stopping_criterion': stopping_criterion_cache,
                                'best_checkpoint': epoch_ref - 1}
                        with open('{}.json'.format(outfile), 'w') as f:
                            json.dump(data, f)
                        return
                except KeyError:
                    pass

            # save intermediate results
            if outfile:
                data = {'losses': self.loss_cache,
                        'stopping_criterion': stopping_criterion_cache}
                with open('{}.json'.format(outfile), 'w') as f:
                    json.dump(data, f)
                self.save_model('{}_{}.cpt'.format(outfile, epoch_ref))

            print('-------------------------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------------------------')

        print('Finished training!')
        print('Ran {} epochs. Final average training losses: {}.'.format(
            max(self.loss_cache), self.loss_cache[max(self.loss_cache.keys())]
        ))
        end = time.time()
        print('Training time: {} seconds'.format(round(end-start, 2)))

    def stopping_criterion(self):

        validation_ranking = self.synonym_retrieval_test(validation=True)
        instances, rankings = zip(*validation_ranking)
        mAP = SynonymRetrieval().mean_average_precision(rankings)

        return mAP

    def synonym_retrieval_test(self, validation=False, baseline=False, outfile=''):

        data = self.sampling.data

        ontology = data['ontology']
        if validation:
            test = data['validation']
        else:
            test = data['test']
        test_pairs = {reference: concept for (concept, reference, positive) in test}

        ranker = SynonymRetrieval()
        ranker.load_ontology(ontology)

        if baseline:
            train_embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        else:
            train_embeddings = self.extract_online_lstm_embeddings(prune=False, normalize=True, verbose=True)

        test_embeddings = deepcopy(train_embeddings)
        ranker.load_train_vectors_object(train_embeddings)
        ranker.load_test_vectors_object(test_embeddings)

        test_ranking = ranker.synonym_retrieval_test(test_pairs, outfile=outfile)

        return test_ranking

    def synonym_retrieval_zeroshot(self, baseline=False, isolated=True, outfile=''):

        data = self.sampling.data

        ontology = data['ontology']
        test_pairs = data['zero-shot']

        ranker = SynonymRetrieval()
        ranker.load_ontology(ontology)

        if baseline:
            train_embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        else:
            train_embeddings = self.extract_online_lstm_embeddings(prune=False, normalize=True, verbose=True)

        test_embeddings = deepcopy(train_embeddings)
        ranker.load_train_vectors_object(train_embeddings)
        ranker.load_test_vectors_object(test_embeddings)

        test_ranking = ranker.synonym_retrieval_zeroshot(test_pairs, isolated=isolated, outfile=outfile)

        return test_ranking

    def synonym_retrieval_train(self, baseline=False, outfile=''):

        data = self.sampling.data

        ontology = data['ontology']
        train = data['train']
        train_pairs = {reference: concept for (concept, reference, positive) in train}

        ranker = SynonymRetrieval()
        ranker.load_ontology(ontology)

        if baseline:
            train_embeddings = deepcopy(self.sampling.pretrained_name_embeddings)
        else:
            train_embeddings = self.extract_online_lstm_embeddings(prune=False, normalize=True, verbose=True)

        ranker.load_train_vectors_object(train_embeddings)

        train_ranking = ranker.synonym_retrieval_train(train_pairs, outfile=outfile)

        return train_ranking
