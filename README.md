# Ubervvald

Ubervvald (Unified toolBench for Edge-targeted impRovements & enVironment-friendly innoVative aLgorithms for DNNs) is a Python library for executing a DNAS and/or Quantization pipeline on PyTorch models. The library is meant to be flexible and "plug-and-play", allowing to either integratie into a larger execution process via only 2-5 function calls or to customize your own pipeline.
 
## Development stage: Alpha 

## System requirements

Tested on: Ubuntu 20.04, ONNXRuntime 1.14, ONNX 1.13, Pytorch 2.3+cu118 or higher (including CUDA) w/ torchvision 0.18 or higher

ONNX models deployed on: Nvidia DRIVEOS 5.2.6 (~Ubuntu 18.04 on aarch64), ONNX OPset 14, ONNXRuntime 1.14

</br></br>

Nevertheless, the package is meant to be OS independent, so if enough users signal proper execution, you may put a pull request.

## Usage
Refer to the examples notebooks (more to be added later).
Ultimately, the package is meant to be flexible by allowing to customize your own pipeline as you see fit.

### Important note:
- DNAS is supported for torch models, while Quantization happens utilizing the ONNX model. There are some functions that handle the transition from one format to another. Additionally, there are other function that require the model in both formats for comparison measurements purpose.

## Credits
This pip package has several reference sources (the package is a derivative work of all the references). Thus I wish to credit all of the authors who made it possible to create this package.
1. PLiNIO's authors (https://github.com/eml-eda/plinio): D.J. Pagliari, M. Risso, B.A. Motetti, A. Burrello @ Politehnico di Torino
2. EFCL Summer School 2024 Track 3 (https://github.com/eml-eda/efcl-school-t3) organizers: S. Benatti, D.J. Pagliari, A. Burrello @ Politehnico di Torino

Additionally, the following sources were used to make the pip package complete:
1. https://pytorch.org/ignite/_modules/ignite/handlers/checkpoint.html#Checkpoint
2. https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
3. https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27
4. All the dependency's API and/or Source Code documentation

## Troubleshooting
1. Q: CUDA out of memory.

    A: Try restarting your system. CUDA does not free memory in certain situations, especially when intrerrupting execution. You may also try utilizing `torch.cuda.empty_cache()` as much as possible.

## License
Ubervvald is mostly licensed under [Apache License 2.0](LICENSE) but there are some portions of code which were required to be sublicensed with the same one [see example](ubervvald/_utils.py#L145).
