
#Metal Performance Shaders/ macos 에서 torch,gpu를 위한 일종의 백엔드
import sys
import contextlib
from functools import lru_cache

import torch
from modules import errors, shared

if sys.platform == "darwin": #macos unix 운영체재
    from modules import mac_specific

if shared.cmd_opts.use_ipex:
    from modules import xpu_specific


def has_xpu() -> bool:
    return shared.cmd_opts.use_ipex and xpu_specific.has_xpu
    # CPU 또는 GPU와는 다른 컴퓨팅 성능을 제공.

def has_mps() -> bool:  #macos의 torch,gpu를 위한 일종의 backend = mps
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string(): #str 변환
    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    if has_xpu():
        return xpu_specific.get_xpu_device_string()

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu or "all" in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc(): #garbage collector

    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() #Inter-Process Communication
            
# 함수는 GPU 메모리를 다른 프로세스로 전송하는 데 사용.
# GPU 텐서를 다른 프로세스로 전송하려면 먼저 텐서를 호스트 메모리로 복사.
# torch.cuda.ipc_collect() 함수는 호스트 메모리에 있는 텐서를 다른 프로세스의 GPU 메모리로 전송.

    if has_mps():
        mac_specific.torch_mps_gc()

    if has_xpu():
        xpu_specific.torch_xpu_gc()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        device_id = (int(shared.cmd_opts.device_id) if shared.cmd_opts.device_id is not None and shared.cmd_opts.device_id.isdigit() else 0) or torch.cuda.current_device()
        if torch.cuda.get_device_capability(device_id) == (7, 5) and torch.cuda.get_device_name(device_id).startswith("NVIDIA GeForce GTX 16"):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu: torch.device = torch.device("cpu")
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


nv_rng = None


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "Unet에서 모든값이 None인 텐서 생성."

        if not shared.cmd_opts.no_half:
            message += "이미지 생성 과정 중 정확도 부족으로 인해 이미지가 정확하게 표현되지 않거나, 그래픽 카드가 특정 half type 데이터 형식을 지원하지 않아 계산 오류가 발생하는 경우 생길 수 있습니다. 정확도를 높이기 위해 특정 계산에 대해 더 정밀한 float32로 설정하세요 in Settings > Stable Diffusion 또는 -no-half commandline argument to fix this."

    elif where == "vae":
        message = "VAE에서 NAN tensor 형성"

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " 이미지 생성 과정 중 정확도 부족으로 인해 이미지가 정확하게 표현되지 않았을 수 있습니다. commandline에 --no-half-vae commandline argument를 붙여넣으세요."
    else:
        message = "A tensor with all NaNs was produced."


    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


@lru_cache #least recently used memory 삭제
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)
