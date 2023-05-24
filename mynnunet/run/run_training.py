import os
import torch
from typing import Union, Optional
import mynnunet
from mynnunet.utilities.find_class_by_name import recursive_find_python_class
from mynnunet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer



def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_class_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed_data: bool = False,
                          device: torch.device = torch.device('cuda')):
    folder = os.path.join(mynnunet.__path__[0], 'training', 'nnUNetTrainer')
    current_module = 'mynnunet.training.nnUNetTrainer'
    # load nnunet class and do sanity checks
    nnunet_trainer = recursive_find_python_class(folder=folder,
                                                 class_name=trainer_class_name,
                                                 current_module=current_module)
    if nnunet_trainer is None:
        raise RuntimeError(
            f'Could not find requested nnunet trainer {trainer_class_name} in '
            f'mynnunet.training.nnUNetTrainer({folder}).'
            f' If it is located somewhere else, please move it there.')
    assert issubclass(nnunet_trainer, nnUNetTrainer), 'The requested nnunet trainer class must inherit from ' \
                                                      'nnUNetTrainer'


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str,
                 fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 device: torch.device = torch.device('cuda')):
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'Unable to convert given value for fold to int: {fold}. fold must be either "all" or an integer')
            raise e

    # has one gpu
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id,
                                           configuration,
                                           fold,
                                           trainer_class_name,
                                           plans_identifier,
                                           use_compressed_data,
                                           device)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser()
    # need to be specified parameters
    parser.add_argument('data_name_or_id', type=str, help='Dataset name or ID to train with')
    parser.add_argument('configuration', type=str, help='Configuration that should be trained')
    parser.add_argument('fold', type=str, help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4')
    # have default value parameters
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help='Use this to set the device the training should run with. cuda/cpu/mps')
    # optional parameters
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default:nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weighs', type=str, required=False, default=None,
                        help='[OPTIONAL] Path to nnUNet checkpoint file to be used as pretrained model. Will only be '
                             'used when actually training. Beta. Use with caution.')
    parser.add_argument('--use_compressed', default=False, action='store_true', required=False,
                        help='[OPTIONAL] If you set this flag the training cases will not be decompressed.')
    parser.add_argument('-npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing.')
    args = parser.parse_args()

    assert args.device in ['cup', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. ' \
                                                  f'Other devices are not tested/supported. Got:{args.device}'

    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(dataset_name_or_id=args.dataset_name_or_id,
                 configuration=args.configuration,
                 fold=args.fold,
                 trainer_class_name=args.tr,
                 plans_identifier=args.p,
                 pretrained_weights=args.pretrained_weights,
                 num_gpus=args.num_gpus,
                 use_compressed_data=args.use_compressed,
                 export_validation_probabilities=args.npz,
                 continue_training=args.c,
                 only_run_validation=args.val,
                 disable_checkpointing=args.disable_checkpointing,
                 device=device)
