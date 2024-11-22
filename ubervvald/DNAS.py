#REF:https://github.com/eml-eda/efcl-school-t3/blob/main/hands_on_1.ipynb
import os
import sys
# import random
import gc
import copy
import glob
import numbers
import pathlib
import operator as op
import numpy as np
import pandas as pd

import torch
import torch.ao.quantization as tq
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassAccuracy, Mean, MultilabelAccuracy
from torchinfo import summary
from ._utils import Checkpoint, EarlyStopping, BaseConfig, config_wrapper, DEBUG

import plinio
from plinio.methods import PIT
from plinio.cost import params

from collections.abc import Sequence
from typing import Union, Optional, Type, Tuple, Callable, Any, IO


class _NASConfigToken(BaseConfig):
    """
    Handler class for NAS configuration. Does exception handling as well for invalid configuration.
    Intended for internal use only.

    :see-also: Initialization is done in `prepare_NAS()`. Read the function's documentation for proper configuration.
    """
    def __init__(self):
        super().__init__()
        #required param
        self.__epochs: int = None
        self.__net_lr: float = None
        self.__nas_lr: float = None
        self.__patience: int = None
        self.__reg_strengths: Sequence = None
        self.__n_class: int = None
        #optional param
        self.__label_smoothing: Optional[float] = None
        self.__weight_decay: Optional[float] = None
        self.__loss_func: Optional[Callable[[Type[Dataset]], 
                                            Type[torch.nn.Module]]] = None
        self.__opt_func: Optional[Callable[[Type[torch.nn.Module]], 
                                           Type[torch.optim.Optimizer]]] = None
        self.__nas_opt_func: Optional[Callable[[Type[plinio.methods.PIT]], 
                                               Tuple[Type[torch.optim.Optimizer], 
                                                     Type[torch.optim.Optimizer]]]] = None
        self.__qat_qconfig: Optional[Type[torch.ao.quantization.QConfig]] = None
        self.__n_work: Optional[int] = None
        self.__multilabel: Optional[bool] = None
        #internal attr
        self.__lambda: float = None
        self._opt_keys.extend([
                            "__label_smoothing", 
                            "__weight_decay",  
                            "__loss_func", 
                            "__opt_func", 
                            "__nas_opt_func", 
                            "__qat_qconfig",
                            "__n_work",
                            "__multilabel",
                            ])
        
        self._inter_keys.extend([
                            "__lambda", 
                            ])

    #Setters and getters. Setters throw warning(optional param) or error(required param) for invalid input.
    #Optional parameters will return default values from getters when not set

    #Required params
    def set_epochs(self, param_epochs: int) -> bool:
        if type(param_epochs) is not int:
            self._raiseError("epochs")
            return False
        self.__epochs = param_epochs #if self.__epochs is None else self.__epochs
        return True

    def get_epochs(self) -> int:
        return self.__epochs

    def set_net_lr(self, param_net_lr: Union[float, Type[torch.Tensor]]) -> bool:
        if type(param_net_lr) not in [float, Type[torch.Tensor]]:
            self._raiseError("learning_rate")
            return False
        self.__net_lr = param_net_lr #if self.__net_lr is None else self.__net_lr
        return True

    def get_net_lr(self) -> Union[float, Type[torch.Tensor]]:
        return self.__net_lr

    def set_nas_lr(self, param_nas_lr: Union[float, Type[torch.Tensor]]) -> bool:
        if type(param_nas_lr) not in [float, Type[torch.Tensor]]:
            self._raiseError("nas_learning_rate")
            return False
        self.__nas_lr = param_nas_lr #if self.__nas_lr is None else self.__nas_lr

    def get_nas_lr(self) -> Union[float, Type[torch.Tensor]]:
        return self.__nas_lr

    def set_patience(self, param_patience: int) -> bool:
        if type(param_patience) is not int:
            self._raiseError("patience")
            return False
        self.__patience = param_patience #if self.__patience is None else self.__patience
        return True

    def get_patience(self) -> int:
        return self.__patience

    def set_reg_strengths(self, param_reg_strengths: Union[Sequence]) -> bool:
        if not (self._isSequence(param_reg_strengths) or self._isSequence(param_reg_strengths[:-1])):
            self._raiseError("reg_strengths")
            return False
        self.__reg_strengths = param_reg_strengths #if self.__reg_strengths is None else self.__reg_strengths
        return True

    def get_reg_strengths(self) -> Union[Sequence]:
        return self.__reg_strengths
    
    def set_n_class(self, param_n_class: int) -> bool:
        if type(param_n_class) is not int:
            self._raiseError("num_classes")
            return False
        self.__n_class = param_n_class #if self.__n_class is None else self.__n_class
        return True

    def get_n_class(self) -> int:
        return self.__n_class
    
    #Optional params or params with default value
    def set_label_smoothing(self, param_label_smoothing: float) -> bool:
        if not (type(param_label_smoothing) is float) or (param_label_smoothing < .0 or param_label_smoothing > 1.):
            self._throwWarning("label_smoothing", 0.0)
            return False
        self.__label_smoothing = param_label_smoothing #if self.__label_smoothing is None else self.__label_smoothing
        return True

    def get_label_smoothing(self) -> float:
        return .0 if self.__label_smoothing is None else self.__label_smoothing
    
    def set_weight_decay(self, param_weight_decay: float) -> bool:
        if type(param_weight_decay) is not float:
            self._throwWarning("weight_decay", 0)
            return False
        self.__weight_decay = param_weight_decay #if self.__weight_decay is None else self.__weight_decay
        return True

    def get_weight_decay(self) -> float:
        return .0 if self.__weight_decay is None else self.__weight_decay

    def set_loss_func(self, param_loss_func: Callable[[Type[Dataset]], Type[torch.nn.Module]]) -> bool:
        if not callable(param_loss_func):
            self._throwWarning("criterion")
            return False
        self.__loss_func = param_loss_func #if self.__loss_func is None  else self.__loss_func
        return True
    
    def get_loss_func(self) -> Callable[[Type[Dataset]], Type[torch.nn.Module]]:
        return get_criterion if self.__loss_func is None else self.__loss_func

    def set_opt_func(self, param_opt_func: Callable[[Type[torch.nn.Module]], Type[torch.optim.Optimizer]]) -> bool:
        if not callable(param_opt_func):
            self._throwWarning("optimizer")
            return False
        self.__opt_func = param_opt_func if self.__opt_func is None else self.__opt_func
        return True

    def get_opt_func(self) -> Callable[[Type[torch.nn.Module]], Type[torch.optim.Optimizer]]:
        return get_optimizer if self.__opt_func is None else self.__opt_func
  
    def set_nas_opt_func(self, param_nas_opt_func: Callable[[Type[torch.nn.Module]], Tuple[Type[torch.optim.Optimizer], Type[torch.optim.Optimizer]]]) -> bool:
        if not callable(param_nas_opt_func):
            self._throwWarning("nas_optimizers")
            return False
        self.__nas_opt_func = param_nas_opt_func #if self.__nas_opt_func is None else self.__nas_opt_func
        return True

    def get_nas_opt_func(self) -> Callable[[Type[torch.nn.Module]], Tuple[Type[torch.optim.Optimizer], Type[torch.optim.Optimizer]]]:
        return get_nas_optimizers if self.__nas_opt_func is None else self.__nas_opt_func
  
    def set_qat_qconfig(self, param_qat_qconfig: Type[torch.ao.quantization.QConfig]) -> bool:
        if not isinstance(param_qat_qconfig, torch.ao.quantization.QConfig):
            self._throwWarning("qat_qconfig", "non-QAT training")
            return False
        self.__qat_qconfig = param_qat_qconfig #if self.__qat_config is None else self.__qat_config
        return True

    def get_qat_qconfig(self) -> Type[torch.ao.quantization.QConfig]:
        return self.__qat_qconfig    
    
    def set_n_work(self, param_n_work: int) -> bool:
        if type(param_n_work) is not int:
            self._throwWarning("num_workers", 4)
            return False
        self.__n_work = param_n_work #if self.__n_work is None else self.__n_work
        return True

    def get_n_work(self) -> int:
        return 4 if self.__n_work is None else self.__n_work
    
    def set_multilabel(self, param_multilabel: bool) -> bool:
        if type(param_multilabel) is not bool:
            self._throwWarning("multilabel", False)
            return False
        self.__multilabel = param_multilabel #if self.param_multilabel is None else self.param_multilabel
        return True

    def get_multilabel(self) -> bool:
        return False if self.__multilabel is None else self.__multilabel

    #Lambda is regularization lambda, utilized internally
    def _set_lambda(self, param_lambda: float) -> None:
        self.__lambda = param_lambda

    def _get_lambda(self) -> float:
        return self.__lambda


def NASConfig(fn):
    """
    Decorator function to add NAS configuration token as global variable.

    Parameters:
        fn: Function refeence intended to be decorated.
    
    Returns: The decorator.
    """
    c = {"NASConfig": _NASConfigToken.getConfig()}
    return config_wrapper(c, fn) #see _utils.config_wrapper for documentation

def prepare_NAS(config_dict: dict) -> bool:
    """
    Function to prepare the configuration singleton token. Will check given configuration and initialize token if no errors were found.

    Parameters:
        config_dict: A configuration mapping of type `dict`. See below a configuration example.

    Return: `True` if config was initialized successfully, otherwise `False`.

    Example of configuration:
    .. code:: python(code)
        config = {
            'epochs': 150,
            'batch_size': 64,
            'learning_rate': 0.0001,
            'nas_learning_rate': 0.00005,
            'patience': 30,
            'reg_strengths': [1e-6, 1e-8, 1],
            'num_classes': 9,
            'model_dir': 'experiments/',
            'label_smoothing': .25,
            'weight_decay': 1e-4,
            'criterion': qNAS.NAS.get_criterion,
            'optimizer': qNAS.NAS.get_optimizer,
            'nas_optimizer': qNAS.NAS.get_nas_optimizers,
            'qat_config': torch.ao.quantization.get_default_qat_config(backend='qnnpack', version=1),
            'last_layer_activation': torch.nn.functional.sigmoid,
            'num_worker': 4,
            'isGPU': True,
            'isMultilabel': False
        }
    """
    NASConfig = _NASConfigToken.getConfig()
    key_to_func = {
        # 'dataset_dir': NASConfig.set_dataset_dir,
        'epochs': NASConfig.set_epochs,
        'batch_size': NASConfig.set_batch_size,
        'learning_rate': NASConfig.set_net_lr,
        'nas_learning_rate': NASConfig.set_nas_lr,
        'patience': NASConfig.set_patience,
        'reg_strengths': NASConfig.set_reg_strengths,
        'num_classes': NASConfig.set_n_class,
        'model_dir': NASConfig.set_model_dir,
        'label_smoothing': NASConfig.set_label_smoothing,
        'weight_decay': NASConfig.set_weight_decay,
        'criterion': NASConfig.set_loss_func,
        'optimizer': NASConfig.set_opt_func,
        'nas_optimizer': NASConfig.set_nas_opt_func,
        'qat_qconfig': NASConfig.set_qat_qconfig,
        'last_layer_activation': NASConfig.set_fin_act_func,
        'num_worker': NASConfig.set_n_work,
        'isGPU': NASConfig.set_gpu,
        'isMultilabel': NASConfig.set_multilabel,
    }

    ign_keys = ["label_smoothing", 
                "weight_decay", 
                "model_dir", 
                "optimizer", 
                "nas_optimizer", 
                "criterion", 
                "last_layer_activation", 
                "qat_qconfig",
                "num_worker",
                "isGPU",
                "isMultilabel",                                   
                ]
    warn_keys = ["lambda", "notified"]
    
    BaseConfig.prepare_config(config_dict, key_to_func, ign_keys, warn_keys)

    return NASConfig.isMinimalConfig()
    
def build_dataloaders(
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset], 
    ) -> Tuple[Type[DataLoader],Type[DataLoader],Type[DataLoader]]:
    """
    Function to create dataloaders from split datasets.

    Parameters:
        train_ds/test_ds/val_ds: torch Dataset given to be inserted into a DataLoader instance.

    Return: torch Dataloader instances for each parameter.
    """
    generator = torch.Generator().manual_seed(23)
    NASConfig = _NASConfigToken.getConfig()
    num_workers = NASConfig.get_n_work()

    train_loader = DataLoader(
        train_ds, batch_size=NASConfig.get_batch_size(), shuffle=True, pin_memory=True, num_workers=num_workers,
        generator=generator, drop_last=True
    )

        
    val_loader = DataLoader(
        val_ds, batch_size=NASConfig.get_batch_size(), shuffle=False, pin_memory=True, num_workers=num_workers,
        generator=generator,
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=NASConfig.get_batch_size(), shuffle=False, num_workers=num_workers,
        generator=generator, 
    )

    return train_loader, val_loader, test_loader

def get_pit_model(original_model: Type[torch.nn.Module], input_shape: Type[torch.Size]) -> Type[plinio.methods.PIT]:
    """
    Function to convert a Pytorch model to PIT model

    Parameters:
        original_model: torch Model given for NAS.
        input_shape: Shape of the input layer of the model.

    Return: the model of type `PIT`, accesible for NAS.
    """
    model_copy = copy.deepcopy(original_model)
    # call the PIT() constructor on model_copy (this ensures that the original model is not touched).
    # Then move the pit model to the training device. Lastly, return the converted model
    pit_model_init = PIT(model_copy, input_shape=input_shape, cost=params, discrete_cost=True, exclude_types=(torch.nn.Sigmoid, ), exclude_names=("model_copy.activate",))
    pit_model_init.to(_NASConfigToken.getConfig().get_device())
    return pit_model_init

def get_criterion(train_ds: Type[Dataset],
                  ) -> Type[torch.nn.Module]:
    """
    Default function to initialize loss function. Will apply label_smoothing if it has been set in the configuration.
    The dataset must have an attribute `Y` or `labels` of `torch.Tensor` type containing all the labels/Y of the dataset.

    Parameters:
        train_ds: torch dataset to calculate the loss weights tensor according to the following formula: :math:`w_class_i = tot_windows / class_i_windows`.

    Return: the loss function, `torch.nn.CrossEntropyLoss`.
    """
    NASConfig = _NASConfigToken.getConfig()
    device = NASConfig.get_device()
    if "dataset" in dir(train_ds.dataset):
        Y = train_ds.dataset.dataset
    else:
        Y = train_ds.dataset
    if "label" in dir(Y):
        Y = Y.label
    elif "Y" in dir(Y):
        Y = Y.Y

    # compute the number of windows belonging to each class
    _, class_frequencies = torch.unique(Y, return_counts=True)
    # derive corresponding loss weights as w_class_i = tot_windows / class_i_windows
    loss_weights = torch.sum(class_frequencies) / (class_frequencies)
    
    # create an instance of the torch.nn.CrossEntropyLoss with weight and label_smoothing and return it 
    crit = torch.nn.CrossEntropyLoss(weight=loss_weights.to(device), label_smoothing=NASConfig.get_label_smoothing())
    return crit

def get_optimizer(model: Type[torch.nn.Module]) -> Type[torch.optim.Optimizer]:
    """
    Default function to initialize train optimizer. Will apply learning_rate and weight_decay if they have been set in the configuration.

    Parameters:
        model: the model of type `PIT` or `torch.nn.Module`.

    Return: the optimizer function, `torch.optim.Adam`.
    """
    NASConfig = _NASConfigToken.getConfig()
    # return an Adam optimizer set to optimize model.parameters() with the correct learning rate
    return torch.optim.Adam(model.parameters(), lr=NASConfig.get_net_lr(), weight_decay=NASConfig.get_weight_decay())

def get_nas_optimizers(model: Type[plinio.methods.PIT]) -> Tuple[Type[torch.optim.Optimizer], Type[torch.optim.Optimizer]]:
    """
    Default function to initialize NAS optimizers. Will apply learning_rate and weight_decay if they have been set in the configuration.
    Will never apply weight_decay for the second optimizer.

    Parameters:
        model: the model of type `PIT`.

    Return: the optimizer functions, both `torch.optim.Adam`.
    """
    NASConfig = _NASConfigToken.getConfig()
    # create and return two optimizers, one standard Adam instance for the net_parameters() and
    # another Adam instance for the nas_parameters(), with two different learning rates (specified in NASConfig)
    # The second optimizer should have weight_decay set to 0 
    net_optimizer = torch.optim.Adam(model.net_parameters(), lr=NASConfig.get_net_lr(), weight_decay=NASConfig.get_weight_decay())
    arch_optimizer = torch.optim.Adam(model.nas_parameters(), lr=NASConfig.get_nas_lr())
    return net_optimizer, arch_optimizer

def train_one_epoch(
        model: Type[torch.nn.Module], 
        final_act_func: Callable[[Type[torch.Tensor]], Type[torch.Tensor]], 
        criterion: Type[torch.nn.Module], 
        optimizer: Type[torch.optim.Optimizer], 
        train_dl: Type[DataLoader], 
        device: Type[torch.device], 
        num_classes: int, 
        isMultilabel: bool, 
        file: Union[str, Type[IO]] = sys.stdout
    ) -> Tuple[Type[np.ndarray], Type[np.ndarray], Type[np.ndarray]]:
    """
    Function that handles training per epoch.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        final_act_func: Function to post-process output vector, example `torch.nn.functional.sigmoid`.
        criterion: The loss function.
        optimizer: The optimizer.
        train_dl: The dataloader for the dataset to be trained on.
        device: The training device.
        num_classes: The total number of classes.
        isMultilabel: Flag to indicate whether the model is for single- or multilbale classification. 
        file: Path or File descriptor to write function output. Default value is `sys.stdout`.

    Return: Average accuracy, average macro accuracy and loss, all of type `numpy.ndarray`.
    """
    # initialize loss and metrics trackers
    avg_loss = Mean(device=device)
    if isMultilabel:
        avg_acc = MultilabelAccuracy(device=device)
        avg_macro_acc = MultilabelAccuracy(device=device, criteria="hamming")
    else:
        avg_acc = MulticlassAccuracy(device=device)
        avg_macro_acc = MulticlassAccuracy(device=device, average='macro', num_classes=num_classes)

    # set the model in training mode
    model.train()
    step = 0
    # iterate over the dataset
    for sample, target in train_dl:
        # move data and labels to the training device
        sample, target = sample.to(device), target.to(device)
        # training iteration: forward, loss, backward, weight update
        output = model(sample)
        output = final_act_func(output)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update loss and metrics
        avg_loss.update(loss)
        avg_acc.update(output, target)
        avg_macro_acc.update(output, target)
        if torch.cuda.is_available():#ref: https://github.com/lenoqt/PyTorch/blob/main/Book-1/Chapter%206%20-%20Torchaudio.ipynb
            if DEBUG and step % 10 == 0:
                startMem = torch.cuda.memory_reserved()
            # for tensor in [sample, target, loss, output]:
            #     tensor.to(torch.device("cpu"))
            del sample, target, loss, output
            gc.collect()
            torch.cuda.empty_cache()
            if DEBUG and step % 10 == 0:
                endMem = torch.cuda.memory_reserved()
        
        if step % 10 == 0:
            # LINE_UP = '\033[1A'
            # LINE_CLEAR = '\x1b[2K'
            memUsage = 'GPU MEM: {:,}B<->{:,}B| '.format(startMem, endMem) if DEBUG else ""
            print(f"{memUsage}Training... Loss: {avg_loss.compute():.2f}, Acc: {avg_acc.compute():.2f}, Macro Acc: {avg_macro_acc.compute():.2f}", end='\r', file=file)
        step += 1
        
    return avg_loss.compute().cpu().numpy(), avg_acc.compute().cpu().numpy(), avg_macro_acc.compute().cpu().numpy()

def evaluate(
        model: Type[torch.nn.Module], 
        final_act_func: Callable[[Type[torch.Tensor]], Type[torch.Tensor]], 
        criterion: Type[torch.nn.Module], 
        eval_dl: Type[DataLoader], 
        device: Type[torch.device], 
        num_classes: int, 
        isMultilabel: bool, 
    ) -> Tuple[Type[np.ndarray], Type[np.ndarray], Type[np.ndarray]]:
    """
    Function to do evaluation on a dataset.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        final_act_func: Function to post-process output vector, example `torch.nn.functional.sigmoid`.
        criterion: The loss function.
        eval_dl: The dataloader for the dataset to be evaluated on.
        device: The evaluation device.
        num_classes: The total number of classes.
        isMultilabel: Flag to indicate whether the model is for single- or multilbale classification.

    Return: Average accuracy, average macro accuracy and loss, all of type `numpy.ndarray`.
    """
    avg_loss = Mean(device=device)
    if isMultilabel:
        avg_acc = MultilabelAccuracy(device=device)
        avg_macro_acc = MultilabelAccuracy(device=device, criteria="hamming")
    else:
        avg_acc = MulticlassAccuracy(device=device)
        avg_macro_acc = MulticlassAccuracy(device=device, average='macro', num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        for sample, target in eval_dl:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            output = final_act_func(output)
            loss = criterion(output, target)
            avg_loss.update(loss)
            avg_acc.update(output, target)
            avg_macro_acc.update(output, target)
            if torch.cuda.is_available():
                del sample, target, loss, output
                gc.collect()
                torch.cuda.empty_cache()
    return avg_loss.compute().cpu().numpy(), avg_acc.compute().cpu().numpy(), avg_macro_acc.compute().cpu().numpy()

def validate(
        model: Type[torch.nn.Module], 
        criterion: Type[torch.nn.Module], 
        eval_dl: Type[DataLoader],
    ) -> Tuple[Type[np.ndarray], Type[np.ndarray], Type[np.ndarray]]:
    """
    Wrapper function for `evaluate()` which will do evaluation on a dataset.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        criterion: The loss function.
        eval_dl: The dataloader for the dataset to be evaluated on.

    Return: Average accuracy, average macro accuracy and loss, all of type `numpy.ndarray` (return values from `evaluate()`).
    """

    NASConfig = _NASConfigToken.getConfig()
    device = NASConfig.get_device()
    num_classes = NASConfig.get_n_class()
    isMultilabel = NASConfig.get_multilabel()
    last_activation_func = NASConfig.get_fin_act_func()
    
    return evaluate(model, last_activation_func, criterion, eval_dl, device, num_classes, isMultilabel)

def nas_train_loop(
        checkpoint_dir: Union[str, Type[pathlib.PurePath], Type[pathlib.Path]], 
        model: Type[torch.nn.Module], 
        train_dl: Type[DataLoader], 
        val_dl: Type[DataLoader], 
        isNAS: bool = False,
        file: Union[str, Type[IO]] = sys.stdout
    ) -> Type[pd.DataFrame]:
    """
    Function that either executes NAS or normal training on a specified trial model. 
    Has checkpoint loading and saving implemented, earlystopping during NAS.

    Parameters:
        checkpoint_dir: Path to the directory to save the checkpoint models in.
        model: the model of base type `torch.nn.Module`.
        train_dl: The dataloader for the dataset to be trained on.
        val_dl: The dataloader for the dataset to be validated on.
        isNAS: Flag to indicate whether its normal training or NAS execution. cWill save history in CSV format if isNAS is False.
        file: Path or File descriptor to write function output. Default value is `sys.stdout`.

    Return: train/NAS history of type `pandas.Dataframe`.
    """
    NASConfig = _NASConfigToken.getConfig()
    device = NASConfig.get_device()
    num_classes = NASConfig.get_n_class()
    isMultilabel = NASConfig.get_multilabel()
    # skip the training if best checkpoint exists.
    if Checkpoint.search_selected(model, checkpoint_dir, device):
        print(f"Skipping {'NAS' if isNAS else 'training'} and loading pre-cooked model from: {checkpoint_dir}")
        return None

    # initialize the two optimizers, with the newly defined function
    if isNAS:
        optimizer, arch_optimizer = NASConfig.get_nas_opt_func()(model)
    else:
        optimizer = NASConfig.get_opt_func()(model)

    criterion = NASConfig.get_loss_func()(train_dl)
    last_activation_func = NASConfig.get_fin_act_func()
    ckp_dict = {"model": model, "optimizer": optimizer}
    checkpoint = Checkpoint(ckp_dict, checkpoint_dir, n_saved=None, comparison_function=op.gt)# if isNAS else op.lt) overfits with loss value
    earlystop = EarlyStopping(patience=NASConfig.get_patience(), toLog=True, comparison_function=op.gt)# if isNAS else op.lt)
    history = []

    # complete this function to return L_task(output, target) + lambda * L_cost
    # remember that L_task is called "criterion", lambda is part of the NASConfig
    # and L_cost is obtained from the PLiNIO class as model.cost
    def nas_criterion(output, target):
        return criterion(output, target) + NASConfig._get_lambda() * model.cost
    
    for epoch in range(NASConfig.get_epochs()):
        if isNAS:
            # set the PLiNIO model to train only the normal weights, and then train it for one epoch on the training set.
            model.train_net_only()

        t_loss, t_acc, t_macro_acc = train_one_epoch(model, last_activation_func, criterion, optimizer, train_dl, device, num_classes, isMultilabel, file)

        if isNAS:
            # set the PLiNIO model to train only the architectural parameters (masks), and then train it for one epoch on the VALIDATION set.
            # Careful: here we need to use a different optimizer, and a different criterion too 
            # i.e., the combined loss defined above
            model.train_nas_only()
            n_loss, n_acc, n_macro_acc = train_one_epoch(model, last_activation_func, nas_criterion, arch_optimizer, val_dl, device, num_classes, isMultilabel, file)
            
        # evaluate the model on the validation set
        v_loss, v_acc, v_macro_acc = evaluate(model, last_activation_func, criterion, val_dl, device, num_classes, isMultilabel)

        if isNAS:
            # log all metrics, as well as the model cost, to the console or to a file.
            # also append these values to the history list as a tuple
            cost = model.cost.detach().cpu().numpy()
            history.append((epoch+1, n_loss, t_loss, t_acc, t_macro_acc, v_loss, v_acc, v_macro_acc, cost))
            print(f"Epoch: {epoch+1}, Total NAS Loss: {n_loss:.2f}, ",
                f"Task Loss: {t_loss:.2f}, Acc: {t_acc:.2f}, Macro Acc: {t_macro_acc:.2f}",
                f"Val Task Loss: {v_loss:.2f}, Val Acc: {v_acc:.2f}, Val Macro Acc: {v_macro_acc:.2f}, ",
                f"Model Cost: {cost:,}", file=file)
            cols = ['epoch','nas_loss', 'loss','acc','macro_acc', 'val_loss', 'val_acc', 'val_macro_acc', 'cost']
        else:
            # logging
            history.append((epoch+1, t_loss, t_acc, t_macro_acc, v_loss, v_acc, v_macro_acc))
            print(f"Epoch: {epoch+1}, Loss: {t_loss:.2f}, Acc: {t_acc:.2f}, Macro Acc: {t_macro_acc:.2f}",
                f"Val Loss: {v_loss:.2f}, Val Acc: {v_acc:.2f}, Val Macro Acc: {v_macro_acc:.2f}", file=file)
            cols = ['epoch','loss','acc','macro_acc', 'val_loss', 'val_acc', 'val_macro_acc']
        # torch.cuda.empty_cache()
        # gc.collect()
        if isNAS and epoch < (NASConfig.get_epochs() - 50):
            # save checkpoint based on the validation macro accuracy.
            # In case of NAS, only in the last 50 epochs, leaving time to the NAS to modify the architecture
            continue
                    
        # save checkpoint
        checkpoint(epoch, v_macro_acc) # v_macro_acc if isNAS else v_loss) overfits with loss value
        # check if we need to early-stop, with the same rationale as checkpointing
        if earlystop(v_macro_acc): # v_macro_acc if isNAS else v_loss):
            print("Stopped at epoch {} because of early stopping".format(epoch + 1))
            break
        

    # create final history dataframe
    history = pd.DataFrame(history, columns=cols)
    # if not isNAS:
    history.to_csv(os.path.join(checkpoint_dir, "history.csv"), index=False)

    # restore the best model and save it as "best.ckp" for later
    checkpoint.load_kbest()
    checkpoint.save_selected(os.path.join(checkpoint_dir, 'best.ckp'))

    return history

def NAS_only(
        model: Type[torch.nn.Module], 
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset]
    ) -> None:
    """
    Function to execute NAS on specified model.
    The function does checks, initializations, saving and loading the model, refreshes dataloaders, records NAS history.
    If `isGPU = True`, will free GPU memory at the start and at the end of each iteration through all regularization strengths specified in the configuration.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        train_dl: The training dataset.
        val_dl: The vaidation dataset.
        test_ds: The test dataset.
    """
    
    NASConfig = _NASConfigToken.getConfig()
    if not NASConfig.isMinimalConfig():
        return
    device = NASConfig.get_device(releaseMem=True)
    model.to(device)
    print(f"Working on: {device}")
    criterion = NASConfig.get_loss_func()(train_ds) # the loss function is defined externally because we need it later

    # add the seed model's macro_acc (on the VALIDATION set) and its number of parameters as the first
    # elements of the two lists
    macro_acc_list = []
    params_list = []
    pit_model = get_pit_model(model, train_ds[0][0].shape)

    train_dl, val_dl, test_dl = build_dataloaders(train_ds, val_ds, test_ds)
    _, _, seed_macro_acc = validate(pit_model, criterion, val_dl)
    seed_size = pit_model.cost.detach().cpu().numpy()
    macro_acc_list.append(seed_macro_acc)
    params_list.append(seed_size)

    # print the seed macro_acc and the initial model cost
    print(f"Val Seed Macro Acc: {seed_macro_acc}")
    print("Model's initial cost (nr. of param): {:,}".format(seed_size))

    for strength in NASConfig.get_reg_strengths()[:-1]:
        print(f"Training with strength {strength}")

        # generate a fresh PIT model in each iteration calling the appropriate function
        trial_model = get_pit_model(model, train_ds[0][0].shape)

        # generate a fresh instance of the dataloaders
        train_dl, val_dl, test_dl = build_dataloaders(train_ds, val_ds, test_ds)
        
        # generate a fresh instance of the loss function
        criterion = NASConfig.get_loss_func()(train_dl)

        # update the training config setting the correct value for lambda
        NASConfig._set_lambda(strength)

        # run the NAS loop defined above, saving the checkpoints under "model_dir/pit_nas_<STRENGTH_VALUE>".
        # you can suppress logging if you want, by setting the file parameter to /dev/null
        history = nas_train_loop(os.path.join(NASConfig.get_model_dir(), f"pit_nas_{strength}"), trial_model, train_dl, val_dl, isNAS=True)

        # perform a last evaluation of the optimized model on the TEST set, 
        # print it to the console, and append the macro accuracy to macro_acc_list. 
        # Also, append the final value of model.cost to the params_list.
        # Validate is just a wrapper function for evaluate
        test_loss, test_acc, test_macro_acc = validate(trial_model, criterion, test_dl)
        print(f"Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}, Test Macro Acc: {test_macro_acc:.2f}")
        macro_acc_list.append(test_macro_acc)
        params_list.append(trial_model.cost.item())
        NASConfig.get_device(releaseMem=True)

def select_NAS_NN(
        model: Type[torch.nn.Module], 
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset]
    ) -> Tuple[Type[torch.nn.Module], float]:
    """
    This function will select regularization strength via keyboard input and load in the model if it was saved as expected.
    Will free GPU memory if `isGPU = True` at the start and at the end of the function.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        train_dl: The training dataset.
        val_dl: The vaidation dataset.
        test_ds: The test dataset.

    Return: the final model of type `torch.nn.Module` on CPU and the selected regularization strength.
    """
    NASConfig = _NASConfigToken.getConfig()
    if not NASConfig.isMinimalConfig():
        return

    if not glob.glob(os.path.join(NASConfig.get_model_dir(), f"pit_nas_*")):
        print(f"\n\n\nERROR:\tCannot find NAS-ed model under ./{os.path.join(NASConfig.get_model_dir(), 'pit_nas_*')}. Be sure to have a directory named pit_nas_<STRENGTH_VALUE>/.\n Cannot continue NAS finetune, thus exiting from execution\n\n\n")
        return

    model.to(NASConfig.get_device(releaseMem=True))
    train_dl, _, test_ds = build_dataloaders(train_ds, val_ds, test_ds)
    criterion = NASConfig.get_loss_func()(train_dl)
    pit_model = get_pit_model(model, train_ds[0][0].shape)

    # define a variable called SELECTED_STRENGTH 
    # with the strength value you want to use for fine-tuning 
    selected_idx = -1
    if isinstance(NASConfig.get_reg_strengths()[-1], numbers.Integral):
        #if the selected strength was pre-configured, select it then.
        selected_idx += NASConfig.get_reg_strengths()[-1] + 1
    else:
        #initiate interactive UI for user to choose the regularization strength via Keyboard
        tries = 0
        idx = 1
        print("The following regularization strength has been found:")
        for strength in NASConfig.get_reg_strengths():
            print(f"\t{idx}. {strength}")
            idx += 1

        while(selected_idx < 0):
            try:
                selected_idx += int(input("Please select a regularization strength:\n>>> "))
                if selected_idx not in list(range(len(NASConfig.get_reg_strengths()))):
                    print("ERROR: \t selected index out of interval,. Please give another index\n")
                    selected_idx = -1
            except:
                if tries < 3:
                    print("ERROR: \t invalid input. Try again\n")
                    tries += 1
                else:
                    print("WARNING: \t too many tries. Defaulting to first regularization strength")
                    selected_idx += 1

    SELECTED_STRENGTH=NASConfig.get_reg_strengths()[selected_idx]
    print(f"Selected regularization strength: {SELECTED_STRENGTH}")

    # load the selected model and evaluate it again on the validation set
    Checkpoint.search_selected(pit_model, os.path.join(NASConfig.get_model_dir(), f"pit_nas_{SELECTED_STRENGTH}"), NASConfig.get_device(releaseMem=True))
    # Validate is just a wrapper function for evaluate
    _, _, test_macro_acc = validate(pit_model, criterion, test_ds)
    print(f"Test Macro Acc after re-loading: {test_macro_acc}")
    print("Model's cost (nr. of param) after NAS: {:,}".format(pit_model.cost.item()))

    # set net and nas parameters to trainable, export the model and move it to the training 
    pit_model.train_net_and_nas()
    ml = pit_model.export()
    ml.to(NASConfig.get_device(releaseMem=True))
    return ml, SELECTED_STRENGTH

def export_NAS(
        model: Type[torch.nn.Module], 
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset]
    ) -> Type[torch.nn.Module]:
    """
    Function to export the model if only NAS is intended to be executed. Should be used in case of not using the library's `NAS_finetune` or `NAS_train` functions.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        train_dl: The training dataset.
        val_dl: The vaidation dataset.
        test_ds: The test dataset.
    
    Return: the exported model of type `torch.nn.Module` ready to be used within the PyTorch framework.
    """
    
    final_model, _ = select_NAS_NN(model, train_ds, val_ds, test_ds)
    final_model.to(_NASConfigToken.getConfig().get_device(isGPU=False))
    torch.save(final_model, _NASConfigToken.getConfig().get_model_dir() / f"pit_nas_extracted.pt")
    return torch.load(_NASConfigToken.getConfig().get_model_dir() / f"pit_nas_extracted.pt")

def NAS_finetune(
        model: Type[torch.nn.Module], 
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset],
        searchPIT: bool = False,
    ) -> Type[torch.nn.Module]:
    """
    This function will train with user selected strength via keyboard input. The training process does not do QAT.
    The function does checks, initializations, saving and loading the model, refreshes dataloaders, records train history.
    Will free GPU memory if `isGPU = True` at the start and at the end of the function.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        train_dl: The training dataset.
        val_dl: The vaidation dataset.
        test_ds: The test dataset.
        searchPIT: Flag to either take the model from parameters or load PIT best checkpoint if available. Default is `False`.

    Return: the final model of type `torch.nn.Module` on CPU.
    """
    NASConfig = _NASConfigToken.getConfig()
    if not NASConfig.isMinimalConfig():
        return
    print(f"Working on: {NASConfig.get_device(releaseMem=True)}")
    if searchPIT:
        final_model, SELECTED_STRENGTH = select_NAS_NN(model, train_ds, val_ds, test_ds)
        if not final_model:
            return
    else:
        final_model = model
        final_model.to(NASConfig.get_device(releaseMem=True))
        SELECTED_STRENGTH = "SEED"
    # generate fresh "dataloaders" and "criterion", and use the nas_train_loops() function to train the optimized model
    train_dl, val_dl, test_ds = build_dataloaders(train_ds, val_ds, test_ds)
    criterion = NASConfig.get_loss_func()(train_dl)

    # prepare QAT if config is present:
    qconfig = NASConfig.get_qat_qconfig()
    
    if hasattr(final_model, "fuse_model"):
        final_model.fuse_model(is_qat=True if qconfig else False)
    elif hasattr(final_model, "fuse_nodes") and final_model.fuse_nodes:
        final_model = tq.fuse_modules(final_model, [final_model.fuse_nodes])

    if qconfig and hasattr(final_model, "set_qconfig"):
        final_model.eval()
        final_model.set_qconfig(qconfig)
        # tq.propagate_qconfig_(final_model, qconfig)
        final_model = tq.prepare_qat(final_model.train())

    # for a while save the checkpoints under "model_dir/pit_finetune_<STRENGTH_VALUE>"
    history = nas_train_loop(os.path.join(NASConfig.get_model_dir(), f'pit_finetune_{SELECTED_STRENGTH}'), final_model, train_dl, val_dl)
    # evaluate the final model on the test set, printing the results. Validate is just a wrapper function for evaluate
    test_loss, test_acc, test_macro_acc = validate(final_model, criterion, test_ds)
    print(f"Final Optimized Model Test Loss: {test_loss:.2f}, Test Acc: {test_acc:.2f}, Test Macro Acc: {test_macro_acc:.2f}")

    # convert to quantized model if config is present:
    if qconfig:
        final_model = tq.convert(final_model.eval())
        
    final_model.to(NASConfig.get_device(isGPU=False))
    torch.save(final_model, os.path.join(NASConfig.get_model_dir(), f'final_model_{SELECTED_STRENGTH}.pt'))
    final_model.to(NASConfig.get_device(releaseMem=True))
    return final_model

def NAS_train(
        model: Type[torch.nn.Module], 
        train_ds: Type[Dataset], 
        val_ds: Type[Dataset], 
        test_ds: Type[Dataset]
    ) -> Type[torch.nn.Module]:
    """
    Function to execute NAS on the model and train afterwards with user selected strength via keyboard input. The training process does not do QAT.
    The function does checks, initializations, saving and loading the model, refreshes dataloaders, records NAS and train history.

    Parameters:
        model: the model of base type `torch.nn.Module`.
        train_dl: The training dataset.
        val_dl: The vaidation dataset.
        test_ds: The test dataset.
    """
    if not _NASConfigToken.getConfig().isMinimalConfig():
        return
    
    train_dl, val_dl, test_dl = build_dataloaders(train_ds, val_ds, test_ds)
    print(f"Training data-loader length: {len(train_dl)} (nr. of batches)")
    print(f"Validation data-loader length: {len(val_dl)} (nr. of batches)")
    print(f"Test data-loader length: {len(test_dl)} (nr. of batches)")

    NAS_only(model, train_ds, val_ds, test_ds)
    print()
    return NAS_finetune(model, train_ds, val_ds, test_ds, searchPIT=True)

