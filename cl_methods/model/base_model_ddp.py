import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)
import math
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from evaluations import eval_FOMC, eval_perplexity
from transformers import GenerationConfig
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.texts = []
        with open(file_path, "r") as f:
            if file_path.endswith(".jsonl"):
                self.texts = [json.loads(line)["text"].strip() for line in f]
            elif file_path.endswith(".json"):
                data = json.load(f)
                if isinstance(data, dict):
                    for _, samples in data.items():
                        for text in samples:
                            clean = text.strip().replace("\n", " ")
                            if clean: 
                                self.texts.append(clean)
                elif isinstance(data, list):
                    for d in data:
                        clean = d.get("text", "").strip().replace("\n", " ")
                        if clean:
                            self.texts.append(clean)
            else:
                self.texts = [line.strip() for line in f if line.strip()]

        self.examples = self.tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_tensors="pt"
        )["input_ids"]

    def __len__(self):
        return self.examples.size(0)

    def __getitem__(self, idx):
        return self.examples[idx]

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args, 
                 lr_scheduler):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.test_task_list = test_task_list
        self.args = args
        self.lr_scheduler = lr_scheduler
        
        # AMP setup
        param_dtype = next(model.parameters()).dtype

        self.use_fp16 = param_dtype == torch.float16
        self.use_bf16 = param_dtype == torch.bfloat16

        self.scaler = GradScaler() if self.use_fp16 else None
        self.amp_dtype = None
        if self.use_fp16:
            self.amp_dtype = torch.float16
        elif self.use_bf16:
            self.amp_dtype = torch.bfloat16

    def eval(self, model, device):

        #if self.tokenizer.pad_token is None:
        #    self.tokenizer.pad_token = self.tokenizer.eos_token
        
        batch_size = 1 
        block_size = 128

        stable_norm = 0.1399 # 0.1500
        shift_norm = 0.1344
        stable_path = "/app/src/Chronoberg/cl_methods/new_eval/stable_1750_1799.json"
        shift_path = "/app/src/Chronoberg/cl_methods/new_eval/shifting_1750_1799.json"
        stable_dataset = TextDataset(stable_path, self.tokenizer, block_size)
        shift_dataset  = TextDataset(shift_path, self.tokenizer, block_size)
        stable_loader = DataLoader(stable_dataset, batch_size=batch_size)
        shift_loader  = DataLoader(shift_dataset, batch_size=batch_size)

        stable_nll, stable_ppl = self.evaluate_perplexity(
            model, stable_loader, self.tokenizer, device,
            temperature=0.95, norm_factor=stable_norm, desc="Stable set"
        )

        shift_nll, shift_ppl = self.evaluate_perplexity(
            model, shift_loader, self.tokenizer, device,
            temperature=0.95, norm_factor=shift_norm, desc="Shift set"
        )

        print(f"Stable Set PPL: {stable_ppl:.2f}")
        print(f"Shift Set PPL: {shift_ppl:.2f}")

    def evaluate_perplexity(self, model, dataloader, tokenizer, device, temperature=0.95, norm_factor=1.0, desc="Evaluating"):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, unit="batch"):
                batch = to_device(batch, device) #batch.to(device) to_device(batch, device)
                logits = model(batch).logits
                logits = logits[:, :-1, :] / temperature 
                labels = batch[:, 1:]
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                total_loss += loss.item()
                total_tokens += (labels != tokenizer.pad_token_id).sum().item()

        avg_nll = total_loss / total_tokens
        avg_nll *= norm_factor
        ppl = math.exp(avg_nll)
        return avg_nll, ppl




    def train_one_task(self, task, i_task, iters):
        device = torch.device("cuda") if self.args.local_rank == -1 else torch.device("cuda", self.args.local_rank)
        assert not (self.use_fp16 and self.use_bf16), "FP16 and BF16 both active!"
        assert (
            (self.use_fp16 and next(self.model.parameters()).dtype == torch.float16)
            or (self.use_bf16 and next(self.model.parameters()).dtype == torch.bfloat16)
            or (not self.use_fp16 and not self.use_bf16)
        )

        #if self.args.local_rank != -1:
        #    torch.cuda.set_device(self.args.local_rank)

        train_dataloader = task #self.train_task_list[task]
        grad_acc_steps = self.args.gradient_accumulation_steps
        #eval_dataloader = self.eval_task_list[task]
        print(f"Training task {task} for {iters} iters", len(train_dataloader), self.args.global_rank)

        # tqdm progress bar
        progress_bar = tqdm(total=iters, leave=True, disable=(self.args.global_rank != 0))

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        #perplexity = self.perplexity_evaluation(self.test_task_list)
        #self.eval(self.model, device)

        for _ in range(1):
            for step, batch in enumerate(train_dataloader):
                #del batch['sources']
                batch = to_device(batch, device)
                #with self.optimizer.sampled_params(train=True):

                if self.amp_dtype is not None:
                    with autocast(device_type="cuda", dtype=self.amp_dtype):
                        outputs = self.model(**batch, use_cache=False)
                        loss = outputs.loss / grad_acc_steps
                else:
                    outputs = self.model(**batch, use_cache=False)
                    loss = outputs.loss / grad_acc_steps
                    #loss.backward()
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()



                if (step + 1) % grad_acc_steps == 0 or (step + 1) == len(train_dataloader):
                    if self.use_fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                # update progress bar per micro-batch
                # ---- logging ----
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Step {step+1}, Loss {loss.item() * grad_acc_steps:.4f}",
                        refresh=True,
                    )
                self.lr_scheduler.step()
                if (step + 1) >= iters:
                    break
            #if os.environ["LOCAL_RANK"] == "0":
            #    print("Starting training for task {}".format(i_task))
            #perplexity = self.perplexity_evaluation(self.test_task_list)
            #if os.environ["LOCAL_RANK"] == "0":
            #    print("Average Epoch: ", epoch_loss)
                
            #self.eval(self.model, device)
            #print("Perplexity on task {} after epoch {}: {}".format(i_task, epoch+1, perplexity))
        progress_bar.close() 

    def train_continual(self):
            print("iters per task: ", self.args.num_train_iters)
            for i_task, (_,task) in enumerate(self.train_task_list.items()):
                self.train_one_task(task, i_task, int(self.args.num_train_iters[i_task]))
                self.reset_optimizer_and_scheduler() 
                self.save_model(i_task)

    def reset_optimizer_and_scheduler(self):
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        self.model, self.args.weight_decay
    )

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps= 1e-8,
            betas=(0.9, 0.95),
        )

        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
        )    
        '''
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
            self.args.learning_rate,
            self.args.adam_beta1,
            self.args.adam_beta2,
            self.args.adam_epsilon,
        )
        
        num_update_steps_per_epoch = len(self.train_task_list[0]) // self.args.gradient_accumulation_steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.args.num_train_epochs[0]) * num_update_steps_per_epoch,
            eta_min=0,
        )
        '''
    def save_model(self, round):
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            save_hf_format(model_to_save, self.tokenizer, self.args, sub_folder=str(round))
            
        print_rank_0('Successfully saving model after round {}'.format(round), self.args.global_rank)