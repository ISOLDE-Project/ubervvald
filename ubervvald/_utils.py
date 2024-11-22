
import copy
import os
import re
import numpy as np
import random
import torch
import operator as op

import warnings
import pathlib
from abc import ABCMeta
from numbers import Integral, Number
from collections.abc import Mapping, MutableMapping, Sequence
from collections import OrderedDict
from typing import NamedTuple, TypedDict, Union, Type, Tuple, Callable, Any, IO, Callable, cast, Mapping, Optional, Dict, List

DEBUG: bool = True
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

def config_wrapper(c, fn):
    """
    Base decorator for giving access to the config token for some externally defined functions.

    Parameters:
        c: A dict containing the config token instance.
        fn: Function reference which will be decorated.

    Returns: the Config decorator
    """
    def wrapped(*args, **kwargs):
        fn_globals = fn.__globals__ #access global vars of function fn
        orig_fn_globals = {key: fn_globals[key] for key in c if key in fn_globals}
        fn_globals.update(c) #add the config
        try:
            result = fn(*args, **kwargs)
        finally:
            fn_globals.update(orig_fn_globals)
        return result
    return wrapped

def set_seed(seed: int = 23) -> None:
    """
    Function to set seeding for everything to maximize reproducibility.

    Parameters:
        seed: An int representing the seed code.
    """
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def _to_cpu_tensor(tensor: Union[Type[np.ndarray], Type[torch.Tensor]]) -> Type[torch.Tensor]:
    """
    Helper function to convert from `numpy.ndarray` to `torch.Tensor` on CPU.
    """
    if not torch.is_tensor(tensor):
        return torch.from_numpy(tensor)
    elif tensor.get_device() != -1:
        return tensor.detach().cpu() if tensor.requires_grad else tensor.cpu()
    return tensor
        
def F_score(output: Union[Type[np.ndarray], Type[torch.Tensor]], label: Union[Type[np.ndarray], Type[torch.Tensor]], config_token: Type["BaseConfig"], threshold: float = 0.5, beta: float = 1.) -> Type[torch.Tensor]: 
    """
    Function to calculate the F-score of the model's performance. Will convert output and label tensors into `torch.Tensor` if needed.
    
    Parameters:
        output: Tensor of type `numpy.ndarray` or `torch.Tensor` containing the output of the model.
        label: Tensor of type `numpy.ndarray` or `torch.Tensor` containing the true labels of each batch.
        config_token: Object of base type `BaseConfig` having the last activation function set.
        threshold: The minimum value to be taken into consideration when calculating probabilities within tensors.
        beta: Number which changes the F-score's formula. Default value is `1`, meaning that the function does F1-score calculation.

    Return: the F-score value. 
    """
    output = _to_cpu_tensor(output)
    output = config_token.get_fin_act_func()(output)
    label = _to_cpu_tensor(label)

    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    adjust = 1e-12 # sometimes F2 is NaN due to recall and precision being 0
    precision = torch.mean(TP / (TP + FP + adjust))
    recall = torch.mean(TP / (TP + FN + adjust))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + adjust)
    return F2.mean(0)

class BaseConfig(metaclass=ABCMeta):

    def __init__(self):
        self.__batch_size: int = None
        self.__model_dir: Optional[Union[Type[pathlib.Path], str]] = None
        self.__fin_act_func: Optional[Callable[[Type[torch.Tensor]], 
                                               Type[torch.Tensor]]] = None
        self.__gpu: Optional[bool] = None
        #configurable default value (can change from required to optional if set)
        self._def_batch_size: Optional[int] = None
        self._def_model_dir: Optional[Union[Type[pathlib.Path], str]] = None
        
        #internal use
        self.__notified: bool = False 
        self._opt_keys: Sequence = [
                                    "__model_dir",
                                    "__fin_act_func",
                                    "__gpu",
                                    ]
        self._inter_keys: Sequence = [
                                        "_opt_keys",
                                        "_INSTANCE", 
                                        "_inter_keys",
                                        "_def_batch_size",
                                        "_def_model_dir", 
                                        "__notified"
                                        ]

    _INSTANCE = None

    @classmethod #TODO: maybe auto generate getters, setters?
    def getConfig(cls) -> Type["BaseConfig"]:
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def _isPath(self, value: Any) -> bool:
        return True if isinstance(value, (pathlib.PurePath, pathlib.Path)) else False

    def _isString(self, value: Any) -> bool:
        return True if type(value) is str else False

    def _isNumericType(self, value: Any) -> bool:
        return True if type(value) in [int, float] else False

    def _isSequence(self, value: Any) -> bool:
        isSameType = lambda struc, typ: all(isinstance(elem, typ) for elem in struc)
        return True if isinstance(value, Sequence) and len(value) >= 0 and (isSameType(value, float) or isSameType(value, int)) else False

    def _throwWarning(self, key: str, value: Optional[str] = None, isMissing: bool = False) -> None:
        warn_str = f"key \x1b[4;33m{key}\x1b[0m {'is missing' if isMissing else 'has invalid value'} from the config. Reverting to overwriteable default " + ('' if isMissing else f"{('value ' + value if value is not None else 'function get_' + key + '()')}")
        warnings.warn(
            warn_str, 
            RuntimeWarning,
            stacklevel=0
            )

    def _raiseError(self, key: str, isMissing: bool = False) -> None:
        # newline = '\n\n\n'
        error_msg = f"key \x1b[4;31m{key}\x1b[0m {'is missing' if isMissing else 'has invalid value to be assigned'} from the config. {'' if isMissing else 'Execution stopped'}"
        if isMissing:
            print(error_msg)
        else:
            raise ValueError(error_msg)
        
    def isMinimalConfig(self) -> bool:
        """
        Function to check if required params were set. Will also check the optional parameters on the first execution.

        Return: `True` if  all the required parameters within the config was initialized successfully, otherwise `False`.
        """
        keys_to_ign = copy.copy(self._opt_keys)
        keys_to_ign.extend(self._inter_keys)
        optional_attr_set = set([attr for attr in self.__dict__.keys() if any(key in attr for key in keys_to_ign)])
        required_attr_set = set(self.__dict__.keys()) - optional_attr_set
        min_req_attr = [key for key in required_attr_set if not str(hex(id(getattr(self, key)))) in str(getattr(self, key))]
        if None in [getattr(self, key) for key in min_req_attr]:
            # print("The following required parameters were not set:")
            for miss_key in [key.split("__")[-1].strip("_") for key in min_req_attr if getattr(self, key) is None]:
                self._raiseError(miss_key.strip("_"), isMissing=True)
            # print()
            raise RuntimeError("\nExecution stopped.")
            # return False

        self.checkDefaultParams()
        return True
    
    def checkDefaultParams(self) -> Union[bool, None]:
        """
        Function to notify revertion to default value

        Return: `True` if all optional parameters within the config was initialized successfully, otherwise `False`.
        """
        if self.__notified:
            return
        self.__notified = True
        optional_attr_set = set([attr for attr in self.__dict__.keys() if any(key in attr for key in self._opt_keys)])
        if None in [getattr(self, key) for key in optional_attr_set]:
            for miss_key in [key.split("__")[-1].strip("_") for key in optional_attr_set if getattr(self, key) is None]:
                self._throwWarning(miss_key.strip("_"), isMissing=True)
            print()
            return False
        return True
    
    #Common param get/set
    def set_batch_size(self, param_batch_size: int) -> bool:
        if type(param_batch_size) is not int:
            self._raiseError("batch_size") if not self._def_batch_size else self._throwWarning("batch_size", str(self._def_batch_size))
            return False
        self.__batch_size = param_batch_size #if self.__batch_size is None else self.__batch_size
        return True

    def get_batch_size(self) -> int:
        return self._def_batch_size if self.__batch_size is None else self.__batch_size
    
    def set_model_dir(self, param_model_dir: Union[str, Type[pathlib.PurePath], Type[pathlib.Path]]) -> bool:
        if not (self._isPath(param_model_dir) or self._isString(param_model_dir)):
            self._throwWarning("model_dir", self._def_model_dir)
            return False
        self.__model_dir = param_model_dir #if self.__model_dir is None else self.__model_dir
        pathlib.Path(self.__model_dir).mkdir(parents=True, exist_ok=True)
        return True

    def get_model_dir(self) -> Union[str, Type[pathlib.PurePath], Type[pathlib.Path]]:
        return (self._def_model_dir, #create directory path for default path
        pathlib.Path(self._def_model_dir).mkdir(parents=True, exist_ok=True))[0] if self.__model_dir is None else self.__model_dir
  
    def set_fin_act_func(self, param_fin_act_func: Callable[[Type[torch.Tensor]], Type[torch.Tensor]]) -> bool:
        if not callable(param_fin_act_func):
            self._throwWarning("final_activation")
            return False
        self.__fin_act_func = param_fin_act_func #if self.__fin_act_func is None else self.__fin_act_func
        return True

    def get_fin_act_func(self) -> Callable[[Type[torch.Tensor]], Type[torch.Tensor]]:
        return (lambda x: x) if self.__fin_act_func is None else self.__fin_act_func
    
    def set_gpu(self, param_gpu: bool) -> bool:
        if type(param_gpu) is not bool:
            self._throwWarning("gpu", False)
            return False
        self.__gpu = param_gpu #if self.__gpu is None else self.__gpu
        return True

    def get_gpu(self) -> bool:
        return False if self.__gpu is None else self.__gpu
    
    # @staticmethod
    def get_device(self, isGPU: Optional[bool] = None, releaseMem: bool = False) -> Type[torch.device]:
        """
        Function to acquire training device. Can free memory if device is GPU.

        Parameters:
            isGPU: If `True`, will try to acquire a GPU else will default to CPU. If `None` will get the flag value from the config.
            releaseMem: If `True`, will free the GPU memory in the case of utilizing it. Default value is `False`.

        Return: the training device according to availability.
        """
        if not isGPU:
            isGPU = self.get_gpu()
        if isGPU and torch.cuda.is_available():
            if releaseMem:
                torch.cuda.empty_cache()
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
        
    @staticmethod
    def prepare_config(input_config: Mapping, key_to_func: Mapping, ign_keys: Sequence, warn_keys: Sequence) -> None:
        """
        Base function to set the configuration token.

        Parameters:
            input_config: A dict or any other Mapping type having the desired configuration
            key_to_func: A dict or any other Mapping type with the set function pointers of the config token, according to the key.
            ign_keys: A list or any other Sequence type containing keys to ignore (~= not required).
            warn_keys: A list or any other Sequence type containing keys for which to throw warnings.
        """
        
        # check config for missing fields
        # utilize set ops to see keys present in key_to_func are not in config_dict  #lastly remove optional or params with default value
        missing_keys = set(key_to_func.keys()).difference(set(input_config.keys())) - set(ign_keys)
        # print(missing_keys)
        if len(missing_keys) != 0:
            raise ValueError(f"NAS execution stopped. Missing the following required parameters: \x1b[4;31m{missing_keys}\x1b[0m")
        
        ret_vals = [key_to_func[key](value) for key, value in input_config.items() if key in key_to_func.keys()]

        [warnings.warn(
            f"\x1b[4;33m{wkey}\x1b[0m notified will be ignored",
            UserWarning,
            stacklevel=0
            ) for wkey in warn_keys if wkey in input_config.keys()]
        # return ret_vals

#<------------------------------------------------------------------------------------------------>
# The following source code which is also delimited appropiately
# has the following license and copyright notice:

# BSD 3-Clause License

# Copyright (c) 2018-present, PyTorch-Ignite team
# Copyright (c) 2024-present, David Nevezi-Strango
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#REF: https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping       v0.5.1 
class EarlyStopping():
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events.

    Parameters:
        patience: Number of events to wait if no improvement and then stop the training.
        score_function: It should be a function taking with variable arguments which processes raw output score, 
            and return a score `float`. An improvement is considered if the comparison_function returns `True` by utilizing the score. Default value is `float()` with two decimal precision.
        comparison_function: It should be a comparison function taking two arguments 
            to compare the last score to the best score recorded, and return a 'bool' according to the result of the comparison_function.
            You may use your own function or one built-in (See: https://docs.python.org/3/library/operator.html#module-operator). Default value is `operator.gt()`.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        cumulative_delta: It `True`, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is `False`.
        toLog: Flag to indicate whether logging via STDOUT is required or not. Default value is `False`.
    """

    _state_dict_all_req_keys = (
        "counter",
        "best_score",
    )

    def __init__(
        self,
        patience: int,
        comparison_function: Callable[..., bool] = op.gt, #==max()
        score_function: Callable[..., float] = lambda x : float(format(x, ".2f")),
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
        toLog: bool = False
    ):
        if not callable(score_function):
            raise TypeError("Argument score_function should be a function.")
        
        if not callable(comparison_function):
            raise TypeError("Argument comparison_function should be a function.")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.__score_function = score_function
        self.__comparison_function = comparison_function
        self.__patience = patience
        self.__min_delta = min_delta
        self.__cumulative_delta = cumulative_delta
        self.__counter = 0
        self.__best_score: Optional[float] = None
        self.__toLog = toLog

    def __call__(self, param_score: Union[int, float]) -> bool:
        score = self.__score_function(param_score)

        if self.__best_score is None:
            self.__best_score = score
        elif not self.__comparison_function(score, self.__score_function(self.__best_score + self.__min_delta)):
            # if score is "less" than best + delta, update score if delta is not considered to be cumulative
            if not self.__cumulative_delta and self.__comparison_function(score, self.__best_score):
                self.__best_score = score
            self.__counter += 1
            if self.__toLog:
                print("EarlyStopping: %i / %i" % (self.__counter, self.__patience), end="\r")
            if self.__counter >= self.__patience:
                print("EarlyStopping: Stop training")
                return True
        else:
            #if score is best above all else
            self.__best_score = score
            self.__counter = 0
        return False

    def state_dict(self) -> Dict[str, float]:
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return {"counter": self.__counter, "best_score": cast(float, self.__best_score)}


    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class with provided state dict data.

        Parameters:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        
        self.__counter = state_dict["counter"]
        self.__best_score = state_dict["best_score"]


#REF: https://pytorch.org/ignite/_modules/ignite/handlers/checkpoint.html#Checkpoint       v0.5.1 
class Checkpoint():
    """Checkpoint handler can be used to periodically save and load objects which have attribute
    ``state_dict/load_state_dict``. 

    Parameters:
        to_save: Dictionary with the objects to save. Objects should have implemented ``state_dict`` and
            ``load_state_dict`` methods. 
        ckp_dir: Directory path to which objects will be saved. Will create directories if needed.
        filename_prefix: Prefix for the file name to which objects will be saved. See Note for details.
        comparison_function: It should be a comparison function taking two arguments 
            to compare the last score to the best score recorded, and return a 'bool' according to the result of the comparison_function.
            You may use your own function or one built-in (See: https://docs.python.org/3/library/operator.html#module-operator). Default value is `operator.gt()`.
        score_function: It should be a function taking with variable arguments which processes raw output score, and return a score `float`. If this is set to `None`, will consider epoch number.
            An improvement is considered if the comparison_function returns `True` by utilizing the score. Default value is `(lambda x: float(x))`.
        score_name: If ``score_function`` not None, it is possible to store its value using
            ``score_name``. If ``score_function`` is None, ``score_name`` can be used alone to define ``score_function``
            as ``Checkpoint.get_default_score_fn(score_name)`` by default.
        n_saved: Number of objects that should be kept on disk. Older files will be removed. If set to
            `None`, all objects are kept.
        filename_pattern: If ``filename_pattern`` is provided, this pattern will be used to render
            checkpoint filenames. If the pattern is not defined, the default pattern would be used. See Note for
            details.
        include_self: Whether to include the `state_dict` of this object in the checkpoint. If `True`, then
            there must not be another object in ``to_save`` with key ``checkpointer``.
        greater_or_equal: if `True`, the latest equally scored model is stored. Otherwise, the first model.
            Default, `False`.

    Note:
        **Warning**: Please, keep in mind that if filename collide with already used one to saved a checkpoint,
        new checkpoint will replace the older one. This means that filename like ``checkpoint.pt`` will be saved
        every call and will always be overwritten by newer checkpoints.

    Note:
        To get the last stored filename, handler exposes attribute ``last_checkpoint``:

        .. code-block:: python

            handler = Checkpoint(...)
            ...
            print(handler.last_checkpoint)
            > checkpoint_12345.pt

    Examples:
        Attach the handler to make checkpoints during training:

        .. code-block:: python

            from ignite.handlers import Checkpoint

            model = ...
            optimizer = ...
            lr_scheduler = ...
            avg_acc = ... (torcheval.Metric)
            handler = Checkpoint(
                to_save, '/tmp/models', n_saved=2
            )
            to_save = {'model': model, 'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

            ...

            for epoch in epochs:
            
                ...

                handler(avg_acc.compute().cpu().numpy(), epoch)
                
                ...

            > ["checkpoint_5.pt", "checkpoint_6.pt", ]

        Attach the handler to an evaluator to save best model during the training
        according to computed validation metric:

        .. code-block:: python

            from ignite.handlers import Checkpoint

            # Setup Accuracy metric computation on evaluator.
            # which will be used to define ``score_function`` automatically.
            # Run evaluation on epoch completed event
            # ...

            to_save = {'model': model}
            handler = Checkpoint(
                to_save, '/tmp/models',
                n_saved=2, filename_prefix='best',
                score_name="accuracy"
            )

            ...

            handler(avg_acc.compute().cpu().numpy(), epoch)
    """

    Item = NamedTuple("Item", [("priority", int), ("filename", str)])
    _state_dict_all_req_keys = ("_saved",)

    def __init__(
        self,
        to_save: Mapping,
        ckp_dir: Union[str, Type[pathlib.Path], Type[pathlib.PurePath]],
        filename_prefix: str = "",
        comparison_function: Callable[..., bool] = op.gt, #==max()
        score_function: Union[Callable, None] = lambda x : float(format(x, ".4f")),
        score_name: Optional[str] = None,
        n_saved: Union[int, None] = 1,
        filename_pattern: Optional[str] = None,
        include_self: bool = False,
    ):
        if not isinstance(to_save, Mapping):
            raise TypeError(f"Argument `to_save` should be a dictionary, but given {type(to_save)}")
        self._check_objects(to_save, "state_dict")

        if include_self:
            if not isinstance(to_save, MutableMapping):
                raise TypeError(
                    f"If `include_self` is True, then `to_save` must be mutable, but given {type(to_save)}."
                )

            if "checkpointer" in to_save:
                raise ValueError(f"Cannot have key 'checkpointer' if `include_self` is True: {to_save}")
            
        if "/" in filename_prefix or "\\" in filename_prefix:
            filename_prefix = re.split("/|\\", filename_prefix)[-1]
            warnings.warn(RuntimeWarning(f"Found illegal characters in filename_prefix. Defaulting to the substring after the last '/' or '\\' character."))

        if score_name is not None and score_function is None:
            score_name = None
            warnings.warn(SyntaxWarning(f"Missing Score function. Cannot assign score name {score_name} with no score function specified. Reverting to epoch-only notation."))

        self.to_save = to_save
        self.ckp_dir = pathlib.Path(ckp_dir)
        self.ckp_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.comparison_function = comparison_function
        self.score_function = score_function
        self.score_name = score_name
        self.n_saved = n_saved
        self.ext = "ckp"
        self.filename_pattern = filename_pattern
        self._saved: List["Checkpoint.Item"] = []
        self.include_self = include_self
        self.epoch = None
    
    def _get_filename_pattern(self) -> str:
        if self.filename_pattern is None:
            filename_pattern = self.setup_filename_pattern(
                with_prefix=len(self.filename_prefix) > 0,
                with_score=self.score_function is not None,
                with_score_name=self.score_name is not None,
                with_epoch=self.epoch is not None,
            )
        else:
            filename_pattern = self.filename_pattern
        return filename_pattern

    def _setup_checkpoint(self) -> Dict[str, Any]: 
        if self.to_save is not None:

            def func(obj: Any, **kwargs: Any) -> Dict:
                return obj.state_dict()

            return {key: func(obj) for key, obj in self.to_save.items()}
        return {}

    def _check_lt_n_saved(self, or_equal: bool = False) -> bool:
        if self.n_saved is None:
            return True
        return len(self._saved) < self.n_saved + int(or_equal)

    def _compare_fn(self, new: Union[int, float]) -> bool:
            return self.comparison_function(new, self._saved[0].priority)

    def __call__(self, param_epoch: int, param_value: Number) -> None:
        self.epoch = param_epoch
        if self.score_function is not None:
            priority = self.score_function(param_value)
            if not isinstance(priority, Number):
                raise ValueError("Output of score_function should be a number")
        else:
            priority = param_epoch

        if self._check_lt_n_saved() or (self._compare_fn(priority) and not param_epoch):
            priority_str = f"{priority}" if isinstance(priority, Integral) else f"{priority:.4f}"

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
                checkpoint = checkpoint[name]

            filename_pattern = self._get_filename_pattern()

            filename_dict = {
                "filename_prefix": self.filename_prefix,
                "ext": self.ext,
                "name": name,
                "score_name": self.score_name,
                "score": priority_str if (self.score_function is not None) else None,
                "epoch": self.epoch,
            }
            filename = filename_pattern.format(**filename_dict)

            metadata = {
                "basename": f"{self.filename_prefix}{'_' * int(len(self.filename_prefix) > 0)}{name}",
                "score_name": self.score_name,
                "priority": priority_str,
            }
            checkpoint["metadata"] = metadata

            try:
                index = list(map(lambda it: it.filename == filename, self._saved)).index(True)
                to_remove = True
            except ValueError:
                index = 0
                to_remove = not self._check_lt_n_saved()

            if to_remove:
                item = self._saved.pop(index)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda it: it[0])


            if self.include_self:
                # Now that we've updated _saved, we can add our own state_dict.
                checkpoint["checkpointer"] = self.state_dict()

            torch.save(checkpoint, os.path.join(self.ckp_dir, filename))

    @property
    def last_checkpoint(self) -> Optional[Union[str, Type[pathlib.Path], Type[pathlib.PurePath]]]:
        if len(self._saved) < 1:
            return None

        return self._saved[-1].filename
    
    def reset(self) -> None:
        """Method to reset saved checkpoint names.

        Use this method if the engine will independently run multiple times:

        """
        self._saved = []

    def reload_objects(self, to_load: Mapping, load_kwargs: Optional[Dict] = None, **filename_components: Any) -> None:
        """Helper method to apply ``load_state_dict`` on the objects from ``to_load``. Filename components such as
        name and score can be configured.

        Parameters:
            to_load: a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            load_kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)
            filename_components: Filename components used to define the checkpoint file path.
                Keyword arguments accepted are `name`, `score` and `epoch`.

        Examples:
            .. code-block:: python

                import tempfile

                import torch

                from ignite.handlers import Checkpoint

                model = torch.nn.Linear(3, 3)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

                to_save = {"model": model, "optimizer": optimizer}

                ...

                to_load = to_save
                # load checkpoint myprefix_checkpoint_40.pt
                checkpoint.reload_objects(to_load=to_load, epoch=40)
        """

        filename_pattern = self._get_filename_pattern()

        checkpoint = self._setup_checkpoint()
        name = "checkpoint"
        if len(checkpoint) == 1:
            for k in checkpoint:
                name = k
        name = filename_components.get("name", default=name)
        score = filename_components.get("score", default=None)
        epoch = filename_components.get("epoch", default=None)

        filename_dict = {
            "filename_prefix": self.filename_prefix,
            "ext": self.ext,
            "name": name,
            "score_name": self.score_name,
            "score": score,
            "epoch": epoch,
        }

        checkpoint_fp = filename_pattern.format(**filename_dict)

        path = os.path.join(self.ckp_dir, checkpoint_fp)

        load_kwargs = {} if load_kwargs is None else load_kwargs

        Checkpoint.load_objects(to_load=to_load, checkpoint=path, **load_kwargs)


    def state_dict(self) -> OrderedDict:
        """Method returns state dict with saved items: list of ``(priority, filename)`` pairs.
        Can be used to save internal state of the class.
        """
        return {"_saved": [(priority, filename) for priority, filename in self._saved]}

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replaces internal state of the class with provided state dict data.

        Parameters:
            state_dict: a dict with "saved" key and list of ``(priority, filename)`` pairs as values.
        """
        # super().load_state_dict(state_dict)
        self._saved = [Checkpoint.Item(priority, filename) for priority, filename in state_dict["_saved"]]
        
    def load_kbest(self, k: int = 1) -> None:
        """Method replaces internal state of the class with provided state dict data.

        Parameters:
            k: an int within the range [1, len(all_checkpoints)] to select one of the checkpoints from all the available ones.
        """
        if self._saved is None:
            raise FileNotFoundError("No checkpoints saves are loaded in!")
        print("Loading checkpoint with score: {}".format(self._saved[k - 1].priority))
        Checkpoint.load_objects(to_load=self.to_save, checkpoint=os.path.join(self.ckp_dir, self._saved[k - 1].filename))

    def save_selected(self, path: Union[str, Type[pathlib.Path], Type[pathlib.PurePath]]) -> None:
        """Method to save internal state of the selected model by `load_kbest()` to the provided path.

        Parameters:
            path: `str`, `PurePath` or `Path` pointing to the saved model to be saved.
        """
        if "metadata" in self.to_save:
            obj_dict = self.to_save["metadata"]
        else:
            obj_dict = {
                'epoch': self.epoch,
            }

        for key, obj in self.to_save.items():
            if hasattr(obj, "state_dict"): #in theory, all obj have the state_dict loaded in.
                obj_dict[key] = obj.state_dict()
                
        torch.save(obj_dict, path)

    @staticmethod
    def _check_objects(objs: Mapping, attr: str) -> None:
        def func(obj: Any, **kwargs: Any) -> None:
            if not hasattr(obj, attr):
                raise TypeError(f"Object {type(obj)} should have `{attr}` method")

        [func(obj) for _, obj in objs.items()]
    
    @staticmethod
    def setup_filename_pattern(
        with_prefix: bool = True, with_score: bool = True, with_score_name: bool = True, with_epoch: bool = True
    ) -> str:
        """Helper method to get the default filename pattern for a checkpoint.

        Parameters:
            with_prefix: If `True`, the ``filename_prefix`` is added to the filename pattern:
                ``{filename_prefix}_{name}...``. Default, `True`.
            with_score: If `True`, ``score`` is added to the filename pattern: ``..._{score}.{ext}``.
                Default, `True`. At least one of ``with_score`` and ``with_global_step`` should be `True`.
            with_score_name: If `True`, ``score_name`` is added to the filename pattern:
                ``..._{score_name}={score}.{ext}``. If activated, argument ``with_score`` should be
                also `True`, otherwise an error is raised. Default, `True`.
            with_global_step: If `True`, ``{with_epoch}`` is added to the
                filename pattern: ``...{name}_{with_epoch}...``.
                At least one of ``with_score`` and ``with_epoch`` should be `True`.

        Examples:
            .. code-block:: python

                from ignite.handlers import Checkpoint

                filename_pattern = Checkpoint.setup_filename_pattern()

                print(filename_pattern)
                > "{filename_prefix}_{name}_{with_epoch}_{score_name}={score}.{ext}"
        """
        filename_pattern = "{name}"

        if not (with_epoch or with_score):
            raise ValueError("At least one of with_score and with_epoch should be True.")

        if with_epoch:
            filename_pattern += "_{epoch:03d}"

        if with_score_name and with_score:
            filename_pattern += "_{score_name}={score}"
        elif with_score:
            filename_pattern += "_{score}"
        elif with_score_name:
            raise ValueError("If with_score_name is True, with_score should be also True")

        if with_prefix:
            filename_pattern = "{filename_prefix}_" + filename_pattern

        filename_pattern += ".{ext}"
        return filename_pattern
    
    @staticmethod
    def search_selected(model: Type[torch.nn.Module], checkpoint_dir: Union[str, Type[pathlib.Path], Type[pathlib.PurePath]], device: Type[torch.device]) -> bool:
        """Method to check whether a NAS execution or training was finalized.
        An algorithm (NAS or train) is considred completed if there is a saved model of the selected checkpoint.

        Parameters:
            model: Model of base type `torch.nn.Module`, of which the weights will be loaded in.
            checkpoint_dir: `str`, `PurePath` or `Path` pointing to the directory of the saved model.
            device: The device which the model should be moved to.

        Return: The Model with the 'best' weights or `None` if no model has been found.
        """
        if os.path.exists(os.path.join(checkpoint_dir, 'best.ckp')):
            saved_info = torch.load(os.path.join(checkpoint_dir, 'best.ckp'), map_location='cpu')
            model.load_state_dict(saved_info['model'])
            model = model.to(device)
            return True
        return False
    
    @staticmethod
    def load_objects(to_load: Mapping, checkpoint: Union[str, Mapping, Type[pathlib.PurePath], Type[pathlib.Path]], **kwargs: Any) -> None:
        """Helper method to apply ``load_state_dict`` on the objects from ``to_load`` using states from ``checkpoint``.

        Parameters:
            to_load: a dictionary with objects, e.g. `{"model": model, "optimizer": optimizer, ...}`
            checkpoint: a path, a string filepath or a dictionary with state_dicts to load, e.g.
                `{"model": model_state_dict, "optimizer": opt_state_dict}`. If `to_load` contains a single key,
                then checkpoint can contain directly corresponding state_dict.
            kwargs: Keyword arguments accepted for `nn.Module.load_state_dict()`. Passing `strict=False` enables
                the user to load part of the pretrained model (useful for example, in Transfer Learning)

        Examples:
            .. code-block:: python

                import tempfile
                from pathlib import Path

                import torch
                from ignite.handlers import Checkpoint

                model = torch.nn.Linear(3, 3)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

                to_save = {"model": model, "optimizer": optimizer}

                ...

                to_load = to_save
                checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                checkpoint = torch.load(checkpoint_fp)
                Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

                # or using a string for checkpoint filepath

                to_load = to_save
                checkpoint_fp = Path(tmpdirname) / 'myprefix_checkpoint_40.pt'
                Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_fp)
        """
        if not isinstance(checkpoint, (Mapping, str, pathlib.PurePath, pathlib.Path)):
            raise TypeError(f"Argument checkpoint should be a string or a dictionary, but given {type(checkpoint)}")

        Checkpoint._check_objects(to_load, "load_state_dict")

        if isinstance(checkpoint, (str, pathlib.PurePath, pathlib.Path)):
            checkpoint_obj = torch.load(checkpoint)
        else:
            checkpoint_obj = checkpoint

        def _load_object(obj: Any, chkpt_obj: Any) -> None:
            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(chkpt_obj, **kwargs)
            else:
                obj.load_state_dict(chkpt_obj)

        if len(to_load) == 1:
            # single object and checkpoint is directly a state_dict
            key, obj = list(to_load.items())[0]
            if key not in checkpoint_obj:
                _load_object(obj, checkpoint_obj)
                return

        [_load_object(objec, checkpoint_obj[key]) for key, objec in to_load.items() if key in checkpoint_obj]

#<------------------------------------------------------------------------------------------------>