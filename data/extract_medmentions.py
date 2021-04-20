"""Extract MedMentions data from the source files"""
from collections import defaultdict
from tqdm import tqdm
from pattern_tokenizer import tokenize
from utils import create_pairs_for_medmentions, create_pairs_for_train, create_pairs_for_zeroshot
import json


if __name__ == "__main__":

    ##################################################################
    #########               PREPROCESS                      ##########
    ##################################################################

    # 1) extract and preprocess all mentions from the corpus

    pmids = defaultdict(set)
    infiles = {'train': 'medmentions/corpus_pubtator_pmids_trng.txt',
               'validation': 'medmentions/corpus_pubtator_pmids_dev.txt',
               'test': 'medmentions/corpus_pubtator_pmids_test.txt'}
    for key, infile in infiles.items():
        with open(infile, 'r') as f:
            for line in f:
                pmid = line.strip()
                pmids[key].add(pmid)
    lookup_pmids = {}
    for key, ids in pmids.items():
        for id in ids:
            lookup_pmids[id] = key

    medmentions_infile = 'medmentions/corpus_pubtator.txt'

    collected_mentions = defaultdict(lambda: defaultdict(list))
    all_cuis = set()

    with open(medmentions_infile, 'r') as f:
        for line in tqdm(f, disable=False):
            fields = line.strip().split(('\t'))
            if len(fields) > 1:
                pmid, _, _, mention, _, cui = fields
                cui = cui[5:]
                label = lookup_pmids[pmid]
                collected_mentions[label][cui].append(mention)
                all_cuis.add(cui)

    # 2) preprocess all mentions and reference terms with tokenization and lowercasing

    preprocessed_mentions = {}
    for label, data in collected_mentions.items():
        cleaned = defaultdict(list)
        for cui, names in tqdm(data.items()):
            for name in names:
                pre = ' '.join(tokenize(name)).lower()
                cleaned[cui].append(pre)
        preprocessed_mentions[label] = cleaned


    ##################################################################
    #########               CREATE DATA                      #########
    ##################################################################

    # 1) create training data

    train_ontology = {k: set(v) for k, v in preprocessed_mentions['train'].items()}
    # remove concepts with more than 20 items
    train_ontology = {k: v for k, v in train_ontology.items() if len(v) <= 20}
    train_pairs = create_pairs_for_train(train_ontology)

    # 2) create validation and test data, divide into few-shot and zero-shot

    # validation
    validation_mentions = {k: set(v) for k, v in preprocessed_mentions['validation'].items()}
    validation_mentions_reference = {}
    validation_zeroshot_mentions_reference = {}
    for cui, names in validation_mentions.items():
        for name in names:
            if cui in train_ontology:
                validation_mentions_reference[name] = cui
            else:
                validation_zeroshot_mentions_reference[name] = cui

    fewshot_validation = create_pairs_for_medmentions(validation_mentions_reference, train_ontology, train=False)

    # test
    test_mentions = {k: set(v) for k, v in preprocessed_mentions['test'].items()}
    test_mentions_reference = {}
    test_zeroshot_ontology = defaultdict(set)
    for cui, names in test_mentions.items():
        for name in names:
            if cui in train_ontology:
                test_mentions_reference[name] = cui
            else:
                test_zeroshot_ontology[cui].add(name)

    fewshot_test = create_pairs_for_medmentions(test_mentions_reference, train_ontology, train=False)
    zeroshot_test = create_pairs_for_zeroshot(test_zeroshot_ontology)

    # 3) save all data
    data = {'complete_ontology': {k: list(v) for k, v in train_ontology.items()},
             'ontology': {k: list(v) for k, v in train_ontology.items()},
             'train': sorted(train_pairs),
             'validation': sorted(fewshot_validation),
             'test': sorted(fewshot_test),
             'zero-shot': sorted(zeroshot_test)
                            }

    outfile = 'medmentions.json'
    with open(outfile, 'w') as f:
        json.dump(data, f)
