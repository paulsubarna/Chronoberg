import json
import os
from utils.utils import print_rank_0, save_hf_format, save_zero_three_model
from model.base_model import CL_Base_Model

class lora(CL_Base_Model):
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)

    def check_lora(self):
        # Print trainable parameters to confirm LoRA adapters are active.
        # Also shows total trainable parameters vs total model parameters.
        trainable_params = [name for name, p in self.model.named_parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_model = sum(p.numel() for p in self.model.parameters())

        print("\n===== LoRA Sanity Check =====")
        print(f"Total model parameters: {total_model:,}")
        print(f"Total trainable parameters: {total_trainable:,}")
        print("Trainable parameters (LoRA adapters) include:")
        for name in trainable_params:
            if "lora" in name.lower():
                print("  ", name)
        print("=============================\n")

    def save_model(self, round):
        # Reset LoRA trainable flags
        """for name, param in self.model.named_parameters():
            if "loranew_" in name:
                param.requires_grad = True
            elif "lora_" in name:
                param.requires_grad = False"""

        # Save model and tokenizer
        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            os.makedirs(peft_model_id, exist_ok=True)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            # Update adapter config for compatibility
            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            adapter_config['r_sum'] = 0
            with open(adapter_config_path, 'w') as f:
                json.dump(adapter_config, f)

            print_rank_0(f'Successfully saved the final model to {peft_model_id}', self.args.global_rank)

        def rename_lora_params(module):
            for child_name, child in module.named_children():
                rename_lora_params(child)  # recursive

            for param_name in list(module._parameters.keys()):
                if "loranew_" in param_name:
                    new_name = param_name.replace("loranew_", "lora_")
                    module._parameters[new_name] = module._parameters.pop(param_name)
                    print(f"Renamed {param_name} → {new_name}")

        # Handle DDP-wrapped model
        #model_to_edit = self.model.module if hasattr(self.model, 'module') else self.model
        #rename_lora_params(model_to_edit)