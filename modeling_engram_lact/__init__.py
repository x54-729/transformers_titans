from .configuration_engram_lact import EngramLaCTConfig, EngramConfig
from .modeling_lact import EngramLaCTModel, EngramLaCTForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

__all__ = ['EngramLaCTConfig', 'EngramConfig', 'EngramLaCTModel', 'EngramLaCTForCausalLM']

AutoConfig.register("engram_lact", EngramLaCTConfig)
AutoModel.register(EngramLaCTConfig, EngramLaCTModel)
AutoModelForCausalLM.register(EngramLaCTConfig, EngramLaCTForCausalLM)