#REF: https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
import os
import numpy as np
from tqdm import tqdm

import torch
import onnx
import onnxruntime as ort
from onnxruntime import quantization
from ._utils import BaseConfig, F_score, config_wrapper

from collections.abc import Mapping
from torch.utils.data import Dataset,  DataLoader
from typing import Union, Optional, Type, Tuple, Callable, Any

class _QuntizationDataReader(quantization.CalibrationDataReader):
    """
    Wrapper Class for pyTorch Datasets to be compatible for ONNXRuntime quantization. Intended for internal use only.

    Parameters: 
        torch_ds: The dataset to be used for calibration.
        batch_size: The size of a batch to be given by the DataLoader
        input_name: The name of the first input layer. Needed for ONNXRuntime.

    """
    def __init__(self, torch_ds: Type[Dataset], batch_size: int, input_name: str):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)

    def get_next(self) -> Optional[Mapping]:
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: self._to_numpy(batch[0])}
        else:
          return None

    def rewind(self) -> None:
        self.enum_data = iter(self.torch_dl)

    @staticmethod
    def _to_numpy(tensor: Type[torch.Tensor]) -> Type[np.ndarray]:
        """
        Helper function to convert from `torch.Tensor` to `numpy.ndarray`.
        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class _QConfigToken(BaseConfig):
    """
    Handler Function to handle the path configuration. Does exception handling as well for invalid configuration.
    Intended for internal use only.

    :see-also: Initialization is done in `prepare_quantize()`. Read the function's documentation for proper configuration.
    """
    def __init__(self):
        super().__init__()
        self.__model_arch: str = None #required
        
        #optinal param
        self.__q_format: Optional[Type[ort.quantization.QuantFormat]] = None
        self.__q_a_type: Optional[Type[ort.quantization.QuantType]] = None 
        self.__q_w_type: Optional[Type[ort.quantization.QuantType]] = None 
        #default value
        self._def_batch_size = 1
        self._def_model_dir = "quant_models/"
        
        #internal use
        self._opt_keys.extend([
                            "__q_format",
                            "__q_a_type", 
                            "__q_w_type", 
                            "__batch_size", 
                            ])
    
    #Setters and getters. Setters throw warning(optional param) or error(required param) for invalid input.
    #Optional parameters will return default values from getters when not set

    #Required params
    def set_arch(self, param_model_arch: str) -> bool:
        if not (self._isPath(param_model_arch) or self._isString(param_model_arch)):
            self._raiseError("model architecture")
            return False
        self.__model_arch = param_model_arch #if self.__dataset_dir is None else self.__dataset_dir
        return True
    
    def __get_arch(self) -> str:
        return self.__model_arch
    
    def set_q_format(self, param_q_format: Type[ort.quantization.QuantFormat]) -> bool:
        if not isinstance(param_q_format, quantization.QuantFormat):
            self._throwWarning("quant_format", "QOperator")
            return False
        self.__q_format = param_q_format #if self.__q_format is None else self.__q_format
        return True
    
    def get_q_format(self) -> Type[ort.quantization.QuantFormat]:
        return quantization.QuantFormat.QOperator if self.__q_format is None else self.__q_format
    
    def set_q_a_type(self, param_q_a_type: Type[ort.quantization.QuantType]) -> bool:
        if not isinstance(param_q_a_type, quantization.QuantType):
            self._throwWarning("quant_a_type", "QUINT8")
            return False
        self.__q_a_type = param_q_a_type #if self.__q_a_type is None else self.__q_a_type
        return True
    
    def get_q_a_type(self) -> Type[ort.quantization.QuantType]:
        return quantization.QuantType.QUInt8 if self.__q_a_type is None else self.__q_a_type
    
    def set_q_w_type(self, param_q_w_type: Type[ort.quantization.QuantType]) -> bool:
        if not isinstance(param_q_w_type, quantization.QuantType):
            self._throwWarning("quant_w_type", "QINT8")
            return False
        self.__q_w_type = param_q_w_type #if self.__q_w_type is None else self.__q_w_type
        return True
    
    def get_q_w_type(self) -> Type[ort.quantization.QuantType]:
        return quantization.QuantType.QInt8 if self.__q_w_type is None else self.__q_w_type

    def get_torch_path(self) -> str:
        return os.path.join(self.get_model_dir(), '{arch}.pt'.format(arch=self.__get_arch()))
    
    def get_onnx_path(self) -> str:
        return os.path.join(self.get_model_dir(), '{arch}.onnx'.format(arch=self.__get_arch()))
    
    def get_prep_path(self) -> str:
        return os.path.join(self.get_model_dir(), '{arch}_prep.onnx'.format(arch=self.__get_arch()))
    
    def get_int8_path(self) -> str:
        return os.path.join(self.get_model_dir(), '{arch}_int8.onnx'.format(arch=self.__get_arch()))

def qConfig(fn):
    """
    Decorator function to add NAS configuration token as global variable.

    Parameters:
        fn: Function refeence intended to be decorated.
    
    Returns: The decorator.
    """
    c = {"qConfig": _QConfigToken.getConfig()}
    return config_wrapper(c, fn) #see _utils.config_wrapper for documentation

def prepare_quantize(
        model: Type[torch.nn.Module], 
        config_dict: dict,
        ) -> None:
    """
    Function to prepare the configuration singleton token. Will check given configuration and initialize token if no errors were found.
    It also saves the model in TorschScript format. If GPU is available, it will free it's memory.

    Parameters:
        model: The model to be converted from PyTorch to ONNX format.
        config_dict: A configuration mapping of type `dict`. See below a configuration example.

    Return: `True` if config was initialized successfully, otherwise `False`.

    Example of configuration:
    .. code:: python(code)
        config = {
            'arch': 'resnet18',
            'quant_format': onnxruntime.quantization.QuantFormat.QOperator,
            'quant_a_type': onnxruntime.quantization.QuantType.QUInt8,
            'quant_w_type': onnxruntime.quantization.QuantType.QInt8,
            'batch_size': 1,
            'model_dir': 'experiments/',
            'last_layer_activation': torch.nn.functional.sigmoid,
            'isGPU': False,
        }
    """
    qConfig = _QConfigToken.getConfig()

    key_to_func = {
        "arch": qConfig.set_arch,
        "quant_format": qConfig.set_q_format,
        "quant_a_type": qConfig.set_q_a_type,
        "quant_w_type": qConfig.set_q_w_type,
        "model_dir": qConfig.set_model_dir,
        "batch_size": qConfig.set_batch_size,
        'last_layer_activation': qConfig.set_fin_act_func,
        "isGPU": qConfig.set_gpu,
    }

    ign_keys = [ 
                "model_dir", 
                "batch_size", 
                "last_layer_activation",
                "quant_format",
                "quant_type",
                "isGPU",                                  
                ]
    warn_keys = ["notified"]
    
    BaseConfig.prepare_config(config_dict, key_to_func, ign_keys, warn_keys)

    if not torch.backends.nnpack.is_available():
        torch.backends.nnpack.set_flags(enabled=False)
    model.to(qConfig.get_device(isGPU=False))
    model_jit = torch.jit.script(model)
    model_jit.save(qConfig.get_torch_path())
    del model, model_jit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return qConfig.isMinimalConfig()

def change_input_batch_dim(model: Type[onnx.ModelProto], param_batch_size: Union[str, int] = 1) -> None:
    """
    Change batch size of given model in ONNX format. Works with symbolic batch size as well.

    Parameters: 
        model: The model to be modified.
        param_batch_size: The input batch size to switch to. Size can be symbolic (ex. `N`) as well.

    """
    batch_dim = param_batch_size
    if type(batch_dim) is not str:
        batch_dim = bytes(param_batch_size)

    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = batch_dim

def export_to_onnx(model_pt: Type[torch.nn.Module], model_shape: Type[torch.Size], opset: int = 14) -> None:
    """
    Function to export a TorchScript model to ONNX format and change its input batch size in onnx.
    It will also set the TorchScript model for evaluation only.

    Parameters: 
        model_pt: The model in PyTorch format.
        model_shape: The first layer's input size without batch size.
        opset: ONNX opset to be converted in. Default value is `14`.
    """
    qConfig = _QConfigToken.getConfig()
    batch_size = qConfig.get_batch_size()
    # C, H, W = model_shape
    shape = (batch_size,) + model_shape
    dummy_in = torch.randn(shape, requires_grad=True)
    model_pt.to(qConfig.get_device(isGPU=False))
    model_pt.eval()

    # export fp32 model to onnx
    torch.onnx.export(  model_pt,                                         # model
                        dummy_in,                                         # model input
                        qConfig.get_onnx_path(),                          # path
                        export_params=True,                               # store the trained parameter weights inside the model file
                        opset_version=opset,                              # the ONNX version to export the model to
                        verbose=False,                                    # enable whether logging onto STDOUT
                        do_constant_folding=True,                         # constant folding for optimization
                        input_names = ['input'],                          # input names
                        output_names = ['output'],                        # output names
                        dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                        'output' : {0 : 'batch_size'}})

    model_onnx = onnx.load(qConfig.get_onnx_path())
    onnx.checker.check_model(model_onnx)
    change_input_batch_dim(model_onnx, batch_size)
    # model_onnx = onnx.version_converter.convert_version(model_onnx, opset)
    onnx.save(model_onnx, qConfig.get_onnx_path())

def create_quant_model(calib_ds: Type[Dataset], input_name: str) -> None:
    """
    Function that quantizes statically the specified model. It does static quantization only.

    Parameters:
        calib_ds: The dataset to be used for calibration during quantization.
        input_name: The name of the first layer's input.

    """
    qConfig = _QConfigToken.getConfig()
    batch_size = qConfig.get_batch_size()
    quantization.shape_inference.quant_pre_process(qConfig.get_onnx_path(), qConfig.get_prep_path(), skip_symbolic_shape=False if type(batch_size) is str else True)


    qdr = _QuntizationDataReader(calib_ds, batch_size=batch_size, input_name=input_name)

    q_static_opts = {"ActivationSymmetric":False,
                    "WeightSymmetric":True}
    if qConfig.get_gpu() and torch.cuda.is_available():
        q_static_opts["ActivationSymmetric"] = True

    _ = quantization.quantize_static(model_input=qConfig.get_prep_path(),
                                                    model_output=qConfig.get_int8_path(),
                                                    calibration_data_reader=qdr,
                                                    quant_format=qConfig.get_q_format(),
                                                    activation_type=qConfig.get_q_a_type(),
                                                    weight_type=qConfig.get_q_w_type(),
                                                    extra_options=q_static_opts)

def qEvaluate(
        ref_sess: Type[ort.InferenceSession], 
        sample_sess: Union[Type[ort.InferenceSession], Type[torch.nn.Module]], 
        dl: Type[DataLoader], 
        isTorchSample: bool = False
        ) -> Tuple[Type[torch.Tensor], Type[torch.Tensor]]:
    """
    Function that compares the F-scores of two models in executing in ONNXRuntime environment.

    Parameters:
        ref_sess: The inference session of the first model. Can only be ONNXRuntime session.
        smaple_sess: The inference session of the second model. Can be either a PyTorch or an ONNXRuntime instance.
        dl: The DataLoader of the dataset to be evaluated on.
        isTorchSample: If `sample_sess` is a PyTorch instance, `isTorchSample` should be `True`. Default value is `False`.

    Return: the reference and the sample model's F-score of type `torch.Tensor`.
    """
    ref_history = list()
    sample_history = list() 
    qConfig = _QConfigToken.getConfig()
    device = qConfig.get_device()
    if isTorchSample:
        sample_sess.to()

    for img_batch, label_batch in tqdm(dl, ascii=True, unit="batches"):

        inputs = {ref_sess.get_inputs()[0].name: _QuntizationDataReader._to_numpy(img_batch)}
        ref_outs = ref_sess.run(None, inputs)[0] #extract from ndarray

        ref_history.append(F_score(ref_outs, label_batch, qConfig))
        if isTorchSample:
            img_batch, label_batch = img_batch.to(device), label_batch.to(device) 

            with torch.no_grad():
                sample_outs = sample_sess(img_batch)

        else:
            sample_outs = sample_sess.run(None, inputs)[0] #extract from ndarray

        sample_history.append(F_score(sample_outs, label_batch, qConfig))



    fscore_ref = torch.stack(ref_history).mean()
    fscore_sample = torch.stack(sample_history).mean()
    print("Fscore ref: ", fscore_ref)
    print("Fscore sample: ", fscore_sample)
    print("\n")

    return fscore_ref, fscore_sample

def quantize(ds: Type[Dataset]) -> None:
    """
    Function to execute static quantization on the model.
    The function does initializations, saving and loading the model.

    Parameters:
        ds: The dataset to be utilized for quantization.

    """
    qConfig = _QConfigToken.getConfig()
    #prepare datasets
    offset = 500
    calib_ds = torch.utils.data.Subset(ds, list(range(offset)))
    val_ds = torch.utils.data.Subset(ds, list(range(offset, offset * 2)))
    dl = torch.utils.data.DataLoader(val_ds, batch_size=qConfig.get_batch_size(), shuffle=False)
    
    model_pt = torch.jit.load(qConfig.get_torch_path())
    model_shape = ds[0][0].shape
    
    export_to_onnx(model_pt, model_shape)

    #initialize ONNXRuntime configuration
    ort_provider = ['CPUExecutionProvider']
    if qConfig.get_gpu() and torch.cuda.is_available():
        model_pt.to(qConfig.get_device())
        ort_provider = ['CUDAExecutionProvider']

    #start ONNXRuntime session for non-q ONNX model
    ort_sess = ort.InferenceSession(qConfig.get_onnx_path(), providers=ort_provider)

    #start ONNXRuntime session for qONNX model
    ort_input_name =  ort_sess.get_inputs()[0].name
    create_quant_model(calib_ds, ort_input_name)
    ort_int8_sess = ort.InferenceSession(qConfig.get_int8_path(), providers=ort_provider)

    print("Evaluating ONNX vs TorchScript")
    qEvaluate(ort_sess, model_pt, dl, isTorchSample=True)
    print("Evaluating ONNX vs qONNX")
    qEvaluate(ort_sess, ort_int8_sess, dl)



