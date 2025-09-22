from model.Regular.EWC import EWC
from model.base_model import CL_Base_Model
from model.lora import lora

Method2Class = {"EWC"      : EWC,
                "base"     : CL_Base_Model,
                "lora"     : lora}  # SeqLoRA

AllDatasetName = ["1750","1800","1850","1900","1950" ]