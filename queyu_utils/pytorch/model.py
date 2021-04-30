"""Utils for pytorch model.
"""

import torch
import torch.jit
import torch.cuda
from torch.nn import Module
import os
import time
import enum
import thop
from typing import Tuple, List, Union, Optional

from ..common.others import get_cur_time_str


class ModelSaveMethod(enum.Enum):
    """Methods to save pytorch model.
    """

    WEIGHT = 0
    """Save the weight (state_dict) of model.
    """

    FULL = 1
    """Save full model.
    """

    JIT = 2
    """Save model after JIT trace.
    """


def save_model(model: torch.nn.Module,
               model_file_path: str,
               input_size: Tuple[int, ...] = None,
               save_method: ModelSaveMethod = ModelSaveMethod.WEIGHT):
    """Save pytorch model.

    Args:
        model: A pytorch model.
        model_file_path: Target model file path.
        input_size: The size of input data of model. Defaults to None.
        save_method: The method to save model. Defaults to :attr:`ModelSaveMethod.WEIGHT`.
    """

    model.eval()

    if save_method == ModelSaveMethod.WEIGHT:
        torch.save(model.state_dict(), model_file_path)
    elif save_method == ModelSaveMethod.FULL:
        torch.save(model, model_file_path)
    elif save_method == ModelSaveMethod.JIT:
        assert input_size is not None, 'JIT save method need a dummy input!'
        device = list(model.parameters())[0].device
        dummy_input = torch.rand(input_size).to(device)
        new_model = torch.jit.trace(model, (dummy_input, ), check_trace=False)
        torch.jit.save(new_model, model_file_path)


def load_model(model_file_path: str, save_method: ModelSaveMethod, device: str) -> Union[dict, torch.nn.Module]:
    """Load pytorch model from disk.

    Args:
        model_file_path: File path of a pytorch model.
        save_method: The method to save model.
        device: Target device for loaded model.

    Returns:
        loaded model.
    """
    if save_method == ModelSaveMethod.WEIGHT:
        return torch.load(model_file_path, map_location=device)
    elif save_method == ModelSaveMethod.FULL:
        return torch.load(model_file_path, map_location=device)
    elif save_method == ModelSaveMethod.JIT:
        return torch.jit.load(model_file_path)


def get_model_size(model: torch.nn.Module) -> int:
    """Get the file size of pytorch model.

    Args:
        model: A pytorch model.

    Returns:
        The file size of the model.
    """
    tmp_model_file_path = './tmp-get-model-size-{}-{}.pt'.format(os.getpid(), get_cur_time_str())
    save_model(model, tmp_model_file_path, None, ModelSaveMethod.WEIGHT)

    model_size = os.path.getsize(tmp_model_file_path)
    os.remove(tmp_model_file_path)

    return model_size


def get_model_latency(model: torch.nn.Module, input_size: Tuple[int, ...], sample_num: int, device: str,
                      warmup: bool = True, return_detail: bool = False) -> Union[float, Tuple[float, List[float]]]:
    """Get the latency (inference time) of pytorch model.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_model_latency(model, (1, 3, 224, 224), 10, 'cuda', warmup=True, return_detail=False)
        0.158232
        >>> get_model_latency(model, (1, 3, 224, 224), 10, 'cuda', warmup=True, return_detail=True)
        (0.158232, [0.152312, 0.162301, ...])

    Args:
        model: A pytorch model.
        input_size: The size of input data of model.
        sample_num: The number of dummy input for testing.
        device: Target device for testing.
        warmup:
            Whether inferring several dummy samples to warm for more accurate result. Defaults to ``True``.
        return_detail:
            If True, return all latency info during testing too, i.e. `(avg_latency, latency_list)`,
            or return average latency only. Defaults to ``False``.

    Returns:
        If ``return_detail`` is True, return all latency info during testing too, i.e. `(avg_latency, latency_list)`,
        or return average latency only.
    """
    dummy_input = torch.rand(input_size).to(device)
    model = model.to(device)
    model.eval()

    latency_list = []

    if warmup:
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

    if device == 'cpu':
        with torch.no_grad():
            for _ in range(sample_num):
                start_time = time.time()
                model(dummy_input)
                latency_list += [time.time() - start_time]

    elif 'cuda' in device:
        with torch.no_grad():
            for _ in range(sample_num):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                model(dummy_input)
                ender.record()
                torch.cuda.synchronize(torch.device(device))
                latency_list += [starter.elapsed_time(ender) / 1000.]

    avg_latency = sum(latency_list) / len(latency_list)
    if return_detail:
        return avg_latency, latency_list
    return avg_latency


def get_model_flops_and_params(model: torch.nn.Module, input_size: Tuple[int]) -> Tuple[int, int]:
    """Get FLOPs and number of parameters of pytorch model.

    Args:
        model: A pytorch model.
        input_size: The size of input data of model.

    Returns:
        FLOPs and number of parameters of the model.
    """
    device = list(model.parameters())[0].device
    dummy_input = torch.rand(input_size).to(device)
    return thop.profile(model, dummy_input, verbose=False)


def get_module(model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
    """Get target module from a pytorch model.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_module(model, 'layer1.0.conv1')
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    Args:
        model: A pytorch model.
        module_name: Target module name, which can be previewed by calling ``model.named_modules()`` in pytorch.

    Returns:
        Target module from model. If it doesn't exist, return ``None``.
    """
    for name, module in model.named_modules():
        if name == module_name:
            return module

    return None


def get_super_module(model: torch.nn.Module, module_name: str) -> Optional[torch.nn.Module]:
    """Get super module of target module from a pytorch model.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_super_module(model, 'layer1.0.conv1')
        BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    Args:
        model: A pytorch model.
        module_name: Target module name, which can be previewed by calling ``model.named_modules()`` in pytorch.

    Returns:
        Super module of target module from model. If it doesn't exist, return ``None``.
    """
    super_module_name = '.'.join(module_name.split('.')[0:-1])
    return get_module(model, super_module_name)


def set_module(model: torch.nn.Module, module_name, module: torch.nn.Module):
    """Replace target module with new module in a pytorch model.

    Args:
        model: A pytorch model.
        module_name: Target module name, which can be previewed by calling ``model.named_modules()`` in pytorch.
        module: Module which will be replaced into model.
    """
    super_module = get_super_module(model, module_name)
    setattr(super_module, module_name.split('.')[-1], module)


def get_ith_layer(model: torch.nn.Module, i: int) -> Optional[torch.nn.Module]:
    """Get i-th layer of pytorch model.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> get_ith_layer(model, 3)
        MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    Args:
        model: A pytorch model.
        i: The index of target layer in model.

    Returns:
        i-th layer of model.
    """
    j = 0
    for module in model.modules():
        if len(list(module.children())) > 0:
            continue
        if j == i:
            return module
        j += 1
    return None


def set_ith_layer(model: torch.nn.Module, i: int, layer: torch.nn.Module):
    """Replace i-th layer of pytorch model with new layer.

    Args:
        model: A pytorch model.
        i: The index of target layer in model.
        layer: New layer which will be replaced into model.
    """
    j = 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if j == i:
            set_module(model, name, layer)
            return
        j += 1


def get_module_conv_layers_name(module: torch.nn.Module) -> List[str]:
    """Get the name list of all convolution layers in pytorch module.

    Args:
        module: Target module.

    Returns:
        The name list of all convolution layers in the module.
    """
    res = []
    for name, m in module.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            res += [name]

    return res


class ModuleActivationCapture:
    """Util for capturing input / output feature map of target module in pytorch model after inferring.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> # intend to get the input/output feature map of the first conv layer of first ResBlock in ResNet18
        >>> module_activation_capture = ModuleActivationCapture(get_module(model, 'layer1.0.conv1'), 'cuda')
        >>> # let model perform inference...
        >>> # get the input/output feature map created during last inference
        >>> module_input, module_output = module_activation_capture.input, module_activation_capture.output
        >>> # remove hook to avoid performance waste
        >>> module_activation_capture.remove()

    Args:
        module: Target module.

    Attributes:
        input (Optional[torch.Tensor]):
            Store the input feature map of target module created during last inference.
            Before first inference it's ``None``.
        output (Optional[torch.Tensor]):
            Store the output feature map of target module created during last inference.
            Before first inference it's ``None``.
    """
    def __init__(self, module: torch.nn.Module):
        self.hook = module.register_forward_hook(self._hook_fn)
        self.input = None
        self.output = None

    def _hook_fn(self, module, module_input, module_output):
        self.input = module_input.detach()
        self.output = module_output.detach()

    def remove(self):
        """Remove the hook after getting the input/output.
        """
        self.hook.remove()


class ModuleActivationCaptureWrapper:
    """Util for capturing input / output feature map of a series of modules in pytorch model after inferring.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> # intend to get the input feature map of the first conv layer and
        >>> # the output feature map of the first bn layer of first ResBlock in ResNet18
        >>> # i.e. regard (layer1.0.conv1, layer1.0.bn1) as a (hyper-) module
        >>> module_activation_capture_wrapper = ModuleActivationCaptureWrapper([
        >>>    ModuleActivationCapture(get_module(model, 'layer1.0.conv1'), 'cuda'),
        >>>    ModuleActivationCapture(get_module(model, 'layer1.0.bn1'), 'cuda'),
        >>> ])
        >>> # let model perform inference...
        >>> # get the input/output feature map created during last inference
        >>> module_input = module_activation_capture_wrapper.input
        >>> module_output = module_activation_capture_wrapper.output
        >>> # remove hook to avoid performance waste
        >>> module_activation_capture_wrapper.remove()

    Args:
        mac_list: A series of :class:`ModuleActivationCapture` which belong to a series of modules.
    """
    def __init__(self, mac_list: List[ModuleActivationCapture]):
        self.mac_list = mac_list

    @property
    def input(self) -> Optional[torch.Tensor]:
        """Store the input feature map of first module created during last inference.
        Before first inference it's ``None``.
        """
        return self.mac_list[0].input

    @property
    def output(self) -> Optional[torch.Tensor]:
        """Store the output feature map of last module created during last inference.
        Before first inference it's ``None``.
        """
        return self.mac_list[-1].output

    def remove(self):
        """Remove the hooks after getting the input/output.
        """
        [la.remove() for la in self.mac_list]


class ModuleLatencyProfiler:
    """Util for profiling module latency (inference time) in pytorch model during inferring.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> # intend to get latency of the first conv layer of first ResBlock in ResNet18
        >>> module_latency_profiler = ModuleLatencyProfiler(get_module(model, 'layer1.0.conv1'))
        >>> # let model perform inference...
        >>> # get the latency measured during last inference
        >>> module_latency = module_latency_profiler.latency
        >>> # remove hook to avoid performance waste
        >>> module_latency_profiler.remove()

    Args:
        module: Target module.

    Attributes:
        latency (Optional[float]):
            Store the latency value of target module created during last inference.
            Before first inference it's ``None``.
    """
    def __init__(self, module: torch.nn.Module):
        self._before_infer_hook = module.register_forward_pre_hook(self._before_hook_fn)
        self._after_infer_hook = module.register_forward_hook(self._after_hook_fn)

        self.latency = None

        self._start_time = None

    def _before_hook_fn(self, module, module_input):
        self._start_time = time.time()

    def _after_hook_fn(self, module, module_input, module_output):
        self.latency = time.time() - self._start_time

    def remove(self):
        """Remove the hook after getting the input/output.
        """
        self._before_infer_hook.remove()
        self._after_infer_hook.remove()


class ModuleLatencyProfilerWrapper:
    """Util for profiling latency of a series of modules in pytorch model during inferring.

    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> # intend to get total latency of the first conv layer and
        >>> # the first bn layer of first ResBlock in ResNet18
        >>> # i.e. regard (layer1.0.conv1, layer1.0.bn1) as a (hyper-) module
        >>> module_latency_profiler_wrapper = ModuleLatencyProfilerWrapper([
        >>>    ModuleLatencyProfiler(get_module(model, 'layer1.0.conv1')),
        >>>    ModuleLatencyProfiler(get_module(model, 'layer1.0.bn1')),
        >>> ])
        >>> # let model perform inference...
        >>> # get the total latency measured during last inference
        >>> module_input = module_latency_profiler_wrapper.latency
        >>> # remove hook to avoid performance waste
        >>> module_latency_profiler_wrapper.remove()

    Args:
        mlp_list: A series of :class:`ModuleLatencyProfiler` which belong to a series of modules.
    """
    def __init__(self, mlp_list: List[ModuleLatencyProfiler]):
        self.mlp_list = mlp_list

    @property
    def latency(self):
        """Store the total latency value of target modules created during last inference.
        Before first inference it's ``None``.
        """
        return sum([tp.latency for tp in self.mlp_list])

    def remove(self):
        """Remove the hooks after getting the input/output.
        """
        [tp.remove() for tp in self.mlp_list]
