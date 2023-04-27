def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_name_or_id', type=str, help='Dataset name or ID to train with')
    parser.add_argument('configuration', type=str, help='Configuration that should be trained')
    parser.add_argument('fold', type=str, help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default:nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weighs', type=str, required=False, default=None,
                        help='[OPTIONAL] Path to nnUNet checkpoint file to be used as pretrained model. Will only be '
                             'used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument()
