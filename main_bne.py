import argparse
from bne import EncoderBNE


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fasttext", type=str,
                        help='Path to the fastText embedding model.',
                        required=True)
    parser.add_argument("--outfile", type=str,
                        help='Output file for the model checkpoints and training stats.',
                        required=True)
    parser.add_argument("--gpu", type=int,
                        help='Index of the GPU used during training.',
                        required=True)
    parser.add_argument("--hidden", type=int, default=4800,
                        help='Size of the hidden layer.')
    parser.add_argument("--layers", type=int, default=1,
                        help='Number of hidden layers.')
    parser.add_argument("--batch", type=int, default=64,
                        help='Batch size. Number of names in a batch.')
    parser.add_argument("--dropout", type=float, default=0.5,
                        help='Dropout rate for training.')
    parser.add_argument("--triplet", type=float, default=0.1,
                        help='Triplet margin for the siamese loss.')
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help='Learning rate.')

    args = parser.parse_args()

    print('Initializing encoder...')
    encoder = EncoderBNE(data_infile='data/medmentions.json',
                         fasttext_model_path=args.fasttext,
                         triplet_margin=args.triplet,
                         hidden_size=args.hidden,
                         num_layers=args.layers,
                         batch_size=args.batch,
                         learning_rate=args.learning_rate,
                         dropout_rate=args.dropout,
                         gpu_index=args.gpu,
                         )
    print('Started training...')
    encoder.train(outfile=args.outfile)
    print('Finished training!')

    # retrieve the best checkpoint, load checkpoint, run baseline and trained model
    assert encoder.best_checkpoint
    epoch_ref = encoder.best_checkpoint
    print('Best model checkpoint: {}'.format(epoch_ref))
    encoder.load_model('{}_{}.cpt'.format(args.outfile, epoch_ref))

    print('RESULTS:')

    print('TEST:')
    print('1) trained DAN:')
    results = encoder.synonym_retrieval_test(baseline=False)
    print(results)
    print('2) baseline:')
    results = encoder.synonym_retrieval_test(baseline=True)
    print(results)

    print('ZERO-SHOT TEST:')
    print('1) trained DAN:')
    results = encoder.synonym_retrieval_zeroshot(baseline=False)
    print(results)
    print('2) baseline:')
    results = encoder.synonym_retrieval_zeroshot(baseline=True)
    print(results)

    print('TRAIN:')
    print('1) trained DAN:')
    results = encoder.synonym_retrieval_train(baseline=False)
    print(results)
    print('2) baseline:')
    results = encoder.synonym_retrieval_train(baseline=True)
    print(results)
