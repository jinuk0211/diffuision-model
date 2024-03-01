import os
import yaml
import torch
from torch import nn
import wandb
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType
)

from .utils import Base, EXPECTED, 
from .utils import create_folder_if_necessary, safe_save, load_or_fail

# pylint: disable=unused-argument
class WarpCore(ABC):  #@상속 받을 클래스 추상화 class
    @dataclass(frozen=True)
    class Config(Base): #settable,mandatory field가 있음
        experiment_id: str = ex
        checkpoint_path: str = 
        output_path: str = 
        checkpoint_extension: str = "safetensors"
        dist_file_subfolder: str = ""
        allow_tf32: bool = True

        wandb_project: str = None
        wandb_entity: str = None

    @dataclass() # not frozen, means that fields are mutable
    class Info(): # not inheriting from Base, because we don't want to enforce the default fields
        wandb_run_id: str = None
        total_steps: int = 0
        iter: int = 0

    @dataclass(frozen=True)
    class Data(Base):
        dataset: Dataset = EXPECTED
        dataloader: DataLoader  = EXPECTED
        iterator: any = EXPECTED

    @dataclass(frozen=True)
    class Models(Base):
        pass

    @dataclass(frozen=True)
    class Optimizers(Base):
        pass

    @dataclass(frozen=True)
    class Schedulers(Base):
        pass

    @dataclass(frozen=True)
    class Extras(Base):
        pass
    # ---------------------------------------
    info: Info
    config: Config

    # FSDP Fully sharded data parallel
    fsdp_defaults = {
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
        "cpu_offload": None,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        "limit_all_gathers": True,
    }
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    # ------------

    # OVERRIDEABLE METHODS
    
    #모델,optimizer 불러오기전에 load
    def setup_extras_pre(self) -> Extras:
        return self.Extras()

    # dataset, dataloader and/or iterator 포함 딕셔너리 반환
  
    @abstractmethod 
    def setup_data(self, extras: Extras) -> Data: 
        raise NotImplementedError("추상메소드입니다 override 해주세요")

   #훈련에 사용될 모든 모델들의 딕셔너리 반환
    @abstractmethod
    def setup_models(self, extras: Extras) -> Models:
        raise NotImplementedError("추상메소드입니다 override 해주세요")

    #훈련에 사용될 모든 optimizer 딕셔너리 반환
    @abstractmethod
    def setup_optimizers(self, extras: Extras, models: Models) -> Optimizers:
        raise NotImplementedError("This method needs to be overriden")

    # 훈련에 사용될 모든 lr 스케쥴러 딕셔너리 반환 OPTIONAL
    def setup_schedulers(self, extras: Extras, models: Models, optimizers: Optimizers) -> Schedulers:
        return self.Schedulers()

    # 모든게 setup된 뒤 불러올 model optimizer 등등
    def setup_extras_post(self, extras: Extras, models: Models, optimizers: Optimizers, schedulers: Schedulers) -> Extras:
        return self.Extras.from_dict(extras.to_dict())

    # 훈련 method
    @abstractmethod
    def train(self, data: Data, extras: Extras, models: Models, optimizers: Optimizers, schedulers: Schedulers):
        raise NotImplementedError("This method needs to be overriden")
    # ------------

    def setup_info(self, full_path=None) -> Info:
        if full_path is None:
            full_path = (f"{self.config.checkpoint_path}/{self.config.experiment_id}/info.json")
        info_dict = load_or_fail(full_path, wandb_run_id=None) or {}
        info_dto = self.Info(**info_dict)
        if info_dto.total_steps > 0 and self.is_main_node:
            print(">>> RESUMING TRAINING FROM ITER ", info_dto.total_steps)
        return info_dto

    def setup_config(self, config_file_path=None, config_dict=None, training=True) -> Config:
        if config_file_path is not None:
            if config_file_path.endswith(".yml") or config_file_path.endswith(".yaml"):
                with open(config_file_path, "r", encoding="utf-8") as file:
                    loaded_config = yaml.safe_load(file)
            elif config_file_path.endswith(".json"):
                with open(config_file_path, "r", encoding="utf-8") as file:
                    loaded_config = json.load(file)
            else:
                raise ValueError("Config file must be either a .yml|.yaml or .json file")
            return self.Config.from_dict({**loaded_config, 'training': training})
        if config_dict is not None:
            return self.Config.from_dict({**config_dict, 'training': training})
        return self.Config(training=training)

    def setup_ddp(self, experiment_id, single_gpu=False):
        if not single_gpu:
            local_rank = int(os.environ.get("SLURM_LOCALID"))
            process_id = int(os.environ.get("SLURM_PROCID"))
            world_size = int(os.environ.get("SLURM_NNODES")) * torch.cuda.device_count()

            self.process_id = process_id
            self.is_main_node = process_id == 0
            self.device = torch.device(local_rank)
            self.world_size = world_size
          # Node : GPU가 달려있는 machine
          # - Rank : process ID
          # - Local Rank : 각 node 내부의 process ID
          # - Global Rank : 전체 node의 입장에서의 process ID
          # - World size : 프로세스 수
          # - World Size(W) : 모든 노드에서 실행되는 총 응용 프로그램 프로세스 수
          # - Local World Size(L) : 각 노드에서 실행되는 프로세스 수
          
          

            dist_file_path = f"{os.getcwd()}/{self.config.dist_file_subfolder}dist_file_{experiment_id}"
            # if os.path.exists(dist_file_path) and self.is_main_node:
            #     os.remove(dist_file_path)

            torch.cuda.set_device(local_rank)
            init_process_group(
                backend="nccl", #NVIDIA Collective Communication Library
                rank=process_id,
                world_size=world_size,
                init_method=f"file://{dist_file_path}",
            )
            print(f"[GPU {process_id}] READY")
        else:
            print("Running in single thread, DDP not enabled.")

    def setup_wandb(self): #weight and biases
        if self.is_main_node and self.config.wandb_project is not None:
            self.info.wandb_run_id = self.info.wandb_run_id or wandb.util.generate_id()
            wandb.init(project=self.config.wandb_project, entity=self.config.wandb_entity, name=self.config.experiment_id, id=self.info.wandb_run_id, resume="allow", config=self.config.to_dict())

            if self.info.total_steps > 0:         #프로세스 id라 보면 됨
                wandb.alert(title=f"Training {self.info.wandb_run_id} resumed", text=f"Training {self.info.wandb_run_id} resumed from step {self.info.total_steps}")
            else:
                wandb.alert(title=f"Training {self.info.wandb_run_id} started", text=f"Training {self.info.wandb_run_id} started")

    # LOAD UTILITIES ----------
    def load_model(self, model, model_id=None, full_path=None, strict=True):
        if model_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{model_id}.{self.config.checkpoint_extension}"
        elif full_path is None and model_id is None:
            raise ValueError(
                "This method expects either 'model_id' or 'full_path' to be defined"
            )
                    
          #         Main Node 역할:
          # 실험 메타데이터 저장: 실험 이름, 설정, 하이퍼파라미터 등을 저장합니다.
          # 모델 및 훈련 데이터 로그: 훈련 과정에서 발생하는 데이터, 모델 체크포인트 등을 저장합니다.
          # 실험 결과 시각화: 저장된 데이터를 기반으로 훈련 과정 및 결과를 시각화합니다.
          # W&B 서버와 통신: 훈련 진행 상황을 W&B 서버에 전송하고, 서버에서 제공하는 기능을 활용합니다.
        checkpoint = load_or_fail(full_path, wandb_run_id=self.info.wandb_run_id if self.is_main_node else None)
        if checkpoint is not None:
            model.load_state_dict(checkpoint, strict=strict) #mismatch 발생하면 error ,False라면 mistmatch 무시
            del checkpoint #메모리 효용성

        return model #pretrained 된 모델

    def load_optimizer(self, optim, optim_id=None, full_path=None, fsdp_model=None):
        if optim_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{optim_id}.pt"
        elif full_path is None and optim_id is None:
            raise ValueError(
                "This method expects either 'optim_id' or 'full_path' to be defined"
            )

        checkpoint = load_or_fail(full_path, wandb_run_id=self.info.wandb_run_id if self.is_main_node else None)
        if checkpoint is not None:
            try:
                if fsdp_model is not None:
                    sharded_optimizer_state_dict = (
                        FSDP.scatter_full_optim_state_dict(  # <---- FSDP
                            checkpoint
                            if (
                                self.is_main_node
                                or self.fsdp_defaults["sharding_strategy"]
                                == ShardingStrategy.NO_SHARD
                            )
                            else None,
                            fsdp_model,
                        )
                    )
                    optim.load_state_dict(sharded_optimizer_state_dict)
                    del checkpoint, sharded_optimizer_state_dict
                else:
                    optim.load_state_dict(checkpoint)
            # pylint: disable=broad-except
            except Exception as e:
                print("!!! Failed loading optimizer, skipping... Exception:", e)

        return optim

    # SAVE UTILITIES ----------
    def save_info(self, info, suffix=""):
      # save_info 함수의 suffix 인수는 정보 (모델 학습 정보 예상)를 JSON 형식으로
      # 저장할 때 파일 이름에 문자열 확장자를 선택적으로 추가하는 데 사용 
        full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/info{suffix}.json"
        create_folder_if_necessary(full_path)
        if self.is_main_node:
            safe_save(vars(self.info), full_path)

    def save_model(self, model, model_id=None, full_path=None, is_fsdp=False):
        if model_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{model_id}.{self.config.checkpoint_extension}"
        elif full_path is None and model_id is None:
            raise ValueError(
                "This method expects either 'model_id' or 'full_path' to be defined"
            )
        create_folder_if_necessary(full_path)
        if is_fsdp:
            with FSDP.summon_full_params(model):
                pass
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, self.fsdp_fullstate_save_policy
            ):
                checkpoint = model.state_dict()
            if self.is_main_node:
                safe_save(checkpoint, full_path)
            del checkpoint
        else:
            if self.is_main_node:
                checkpoint = model.state_dict()
                safe_save(checkpoint, full_path)
                del checkpoint

    def save_optimizer(self, optim, optim_id=None, full_path=None, fsdp_model=None):
        if optim_id is not None and full_path is None:
            full_path = f"{self.config.checkpoint_path}/{self.config.experiment_id}/{optim_id}.pt"
        elif full_path is None and optim_id is None:
            raise ValueError(
                "This method expects either 'optim_id' or 'full_path' to be defined"
            )
        create_folder_if_necessary(full_path)
        if fsdp_model is not None:
            optim_statedict = FSDP.full_optim_state_dict(fsdp_model, optim)
            if self.is_main_node:
                safe_save(optim_statedict, full_path)
            del optim_statedict
        else:
            if self.is_main_node:
                checkpoint = optim.state_dict()
                safe_save(checkpoint, full_path)
                del checkpoint
    # -----

    def __init__(self, config_file_path=None, config_dict=None, device="cpu", training=True):
        # Temporary setup, will be overriden by setup_ddp if required
        self.device = device
        self.process_id = 0
        self.is_main_node = True
        self.world_size = 1
        # ----

        self.config: self.Config = self.setup_config(config_file_path, config_dict, training)
        self.info: self.Info = self.setup_info()

    def __call__(self, single_gpu=False):
        # cuda rank로 device를 바꿈, ddp = Distributed Data Parallel
        self.setup_ddp(self.config.experiment_id, single_gpu=single_gpu)  
        self.setup_wandb()
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.is_main_node:
            print()
            print("**STARTIG JOB WITH CONFIG:**")
            print(yaml.dump(self.config.to_dict(), default_flow_style=False))
            print("------------------------------------")
            print()
            print("**INFO:**")
            print(yaml.dump(vars(self.info), default_flow_style=False))
            print("------------------------------------")
            print()

        # SETUP STUFF
        extras = self.setup_extras_pre()
        assert extras is not None, "setup_extras_pre() must return a DTO"

        data = self.setup_data(extras)
        assert data is not None, "setup_data() must return a DTO"
        if self.is_main_node:
            print("**DATA:**")
            print(yaml.dump({k:type(v).__name__ for k, v in data.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        models = self.setup_models(extras)
        assert models is not None, "setup_models() must return a DTO"
        if self.is_main_node:
            print("**MODELS:**")
            print(yaml.dump({
                k:f"{type(v).__name__} - {f'trainable params {sum(p.numel() for p in v.parameters() if p.requires_grad)}' if isinstance(v, nn.Module) else 'Not a nn.Module'}" for k, v in models.to_dict().items()
            }, default_flow_style=False))
            print("------------------------------------")
            print()

        optimizers = self.setup_optimizers(extras, models)
        assert optimizers is not None, "setup_optimizers() must return a DTO"
        if self.is_main_node:
            print("**OPTIMIZERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in optimizers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        schedulers = self.setup_schedulers(extras, models, optimizers)
        assert schedulers is not None, "setup_schedulers() must return a DTO"
        if self.is_main_node:
            print("**SCHEDULERS:**")
            print(yaml.dump({k:type(v).__name__ for k, v in schedulers.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()

        post_extras =self.setup_extras_post(extras, models, optimizers, schedulers)
        assert post_extras is not None, "setup_extras_post() must return a DTO"
        extras = self.Extras.from_dict({ **extras.to_dict(),**post_extras.to_dict() })
        if self.is_main_node:
            print("**EXTRAS:**")
            print(yaml.dump({k:f"{v}" for k, v in extras.to_dict().items()}, default_flow_style=False))
            print("------------------------------------")
            print()
        # -------

        # TRAIN
        if self.is_main_node:
            print("**TRAINING STARTING...**")
        self.train(data, extras, models, optimizers, schedulers)

        if single_gpu is False:
            barrier()
            destroy_process_group()
        if self.is_main_node:
            print()
            print("------------------------------------")
            print()
            print("**TRAINING COMPLETE**")
            if self.config.wandb_project is not None:
                wandb.alert(title=f"Training {self.info.wandb_run_id} finished", text=f"Training {self.info.wandb_run_id} finished")
