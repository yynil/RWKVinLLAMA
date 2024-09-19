from functools import partial
from src.model import RWKV_Tmix_x060, RWKV_CMix_x060,Block
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator
from pytorch_lightning.strategies import DeepSpeedStrategy
# from adam_mini import Adam_mini
import cupy as cp
from cupy.cuda import nccl
import logging
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
import os
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class RWKVDecoderLayer(nn.Module):
    def __init__(
        self,
        args,
        layer_idx: int
    ):
        super(RWKVDecoderLayer, self).__init__()
        self.block = Block(args,layer_idx)
        self.layer_idx = layer_idx
        self.args = args

    def forward(self, hidden_states: torch.Tensor, inference_params=None, *args, **kwargs):
        # Ensure hidden_states requires gradient
        hidden_states.requires_grad_(True)
        if self.args.grad_cp == 1:
            hidden_states = deepspeed.checkpointing.checkpoint(self.block, hidden_states)
        else:
            hidden_states = self.block(hidden_states)
        # hidden_states = self.block(hidden_states)
        # logging.info(f'forward in {self.layer_idx}')
        # so here is just to be compatible with Transformer

        past_key_value = kwargs.get("past_key_value", None)

        if past_key_value is not None:
            dummy_keys = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            dummy_values = torch.ones(
                1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
            )
            # Update kv cache with dummy values
            past_key_value.update(dummy_keys, dummy_values, self.layer_idx)

        return (hidden_states, None, past_key_value)
    
class TimeMixWrapper(nn.Module):
    def __init__(self,args,layer_idx):
        super(TimeMixWrapper, self).__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.time_mixer = RWKV_Tmix_x060(args,layer_idx)
        
    def forward(self,
                hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,):
        args = self.args
        hidden_states.requires_grad_(True)
        if args.grad_cp == 1:
            x = deepspeed.checkpointing.checkpoint(self.time_mixer, hidden_states)
        else:
            x = self.time_mixer(hidden_states)
        return x,None,None


class HybridModel(pl.LightningModule):
    def __init__(self,transformer_model,rwkv_args,teacher_model = None,tokenizer=None):
        super(HybridModel, self).__init__()
        attn_num_heads = transformer_model.config.num_attention_heads
        attn_num_key_value_heads = transformer_model.config.num_key_value_heads
        assert attn_num_heads % attn_num_key_value_heads == 0
        n_share = attn_num_heads // attn_num_key_value_heads
        def init_block_params(rwkv_args,layer_idx,llama_layer):
            if rwkv_args.is_rwkv_att_only:
                decoder = llama_layer
                att = TimeMixWrapper(rwkv_args,layer_idx)
                att.time_mixer.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                att.time_mixer.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                att.time_mixer.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                att.time_mixer.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                del llama_layer.self_attn
                llama_layer.self_attn = att
                return decoder
            else:
                decoder = RWKVDecoderLayer(rwkv_args,layer_idx)
                decoder.block.att.receptance.weight.data = llama_layer.self_attn.q_proj.weight.data
                decoder.block.att.key.weight.data = llama_layer.self_attn.k_proj.weight.data.repeat(n_share, 1)
                decoder.block.att.value.weight.data = llama_layer.self_attn.v_proj.weight.data.repeat(n_share, 1)
                decoder.block.att.output.weight.data = llama_layer.self_attn.o_proj.weight.data
                if rwkv_args.is_llama_ffn:
                    decoder.block.ffn = llama_layer.mlp
                else:
                    decoder.block.ffn.key.weight.data = llama_layer.mlp.up_proj.weight.data
                    decoder.block.ffn.value.weight.data = llama_layer.mlp.down_proj.weight.data
                return decoder
        for layer_idx in range(transformer_model.config.num_hidden_layers):
            if layer_idx in rwkv_args.layers:
                decoder = init_block_params(rwkv_args,layer_idx,transformer_model.model.layers[layer_idx])
                transformer_model.model.layers[layer_idx] = decoder
        self.model = transformer_model
        self.args = rwkv_args
        self.teacher_model = teacher_model
        #free the teacher model
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            if 'pad_token_id' not in self.tokenizer.__dict__:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    def forward(
        self,
        input_ids,
        inference_params=None,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)
    
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # logging.info('decay', lr_decay)
        # logging.info('1x', lr_1x)
        # logging.info('2x', lr_2x)
        # logging.info('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if args.optim=='adam_mini':
                return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if args.optim=='adam_mini':
                return Adam_mini(self, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, weight_decay=0, model_sharding=True, n_feature=args.n_embd, n_head=args.n_embd//64, lora_r=8)
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    def on_fit_start(self):
        args = self.args
        if not args.is_sft and args.teacher_client_mode:
            logging.info('start to initialize process group')
            rank = args.rank
            world_size = (args.world_size//args.num_groups)+1
            cp.cuda.Device(rank).use()
            group_id = rank // (world_size-1)
            logging.info(f'global rank {rank} is in group {group_id} with world size {world_size}')
            nccl_file = f'{args.nccl_file}_{group_id}'
            with open(nccl_file, 'r') as f:
                print(f'load nccl_id from {nccl_file}')
                import json
                nccl_id = json.load(f)['nccl_id']
                args.nccl_id = tuple(nccl_id)
                print("NCCL ID:", nccl_id)
            self.stream = cp.cuda.Stream(non_blocking=True)
            
            # cp.cuda.Device(args.rank).use()
            args.server_rank = world_size-1
            rank = rank % (world_size-1)
            self.recv_buffer = cp.empty((args.micro_bsz, args.max_seq_length,self.model.config.vocab_size), dtype=cp.float32)
            if args.is_hidden_align:
                self.teacher_hidden_states_buffer = cp.empty((args.micro_bsz*self.model.config.num_hidden_layers, args.max_seq_length, self.model.config.hidden_size), dtype=cp.float32)

            logging.info(f'init process group,local rank is {rank} with world size {world_size}, nccl_id is {args.nccl_id}')
            self.comm = nccl.NcclCommunicator(world_size, args.nccl_id, rank)
            logging.info(f'finish init process group, local rank is {rank}')
            
    def validation_step(self, batch, batch_idx):
        args = self.args
        teacher_model = self.teacher_model
        tokenizer = self.tokenizer
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = torch.ne(input_ids, tokenizer.eos_token_id).to(input_ids.device)
        
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,use_cache=False)
        loss = outputs.loss
        
        # 计算perplexity
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        
        if not args.is_sft:
            if args.teacher_client_mode:
                #get teacher logits
                self.comm.send(input_ids.data_ptr(), input_ids.size(0)*input_ids.size(1), nccl.NCCL_INT64, args.server_rank, self.stream.ptr)
                self.stream.synchronize()
                self.comm.recv(self.recv_buffer.data.ptr, self.recv_buffer.size, nccl.NCCL_FLOAT, args.server_rank, self.stream.ptr)
                self.stream.synchronize()
                teacher_logits = torch.as_tensor(self.recv_buffer, device=input_ids.device, dtype=torch.float32)
                logging.info(f'rank {args.rank} is receiving teacher_logits from server, shape is {teacher_logits.shape}')
                if args.is_hidden_align:
                    logging.info(f'rank {args.rank} is receiving teacher_hidden_states from server')
                    self.comm.recv(self.teacher_hidden_states_buffer.data.ptr, self.teacher_hidden_states_buffer.size, nccl.NCCL_FLOAT, args.server_rank, self.stream.ptr)
                    self.stream.synchronize()
                    logging.info(f'rank {args.rank} is receiving teacher_hidden_states from server, shape is {self.teacher_hidden_states_buffer.shape}')
            else:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False,output_hidden_states=args.is_hidden_align)
                teacher_logits = teacher_outputs.logits
                if args.is_hidden_align:
                    teacher_hidden_states = teacher_outputs.hidden_states
            #calculate teacher's loss and perplexity
            # print(f'teacher_logits shape is {teacher_logits.shape}, labels shape is {labels.shape}')

            # 调整 teacher_logits 和 labels 的形状
            teacher_logits_reshaped = teacher_logits.view(-1, teacher_logits.size(-1))  # [batch_size * sequence_length, vocab_size]
            labels_reshaped = labels.view(-1)  # [batch_size * sequence_length]

            # 计算 loss
            teacher_loss = F.cross_entropy(teacher_logits_reshaped, labels_reshaped)

            # 计算 perplexity
            teacher_perplexity = torch.exp(teacher_loss)

            self.log('val_teacher_loss', teacher_loss, prog_bar=True)
            self.log('val_teacher_perplexity', teacher_perplexity, prog_bar=True)
            return {'loss': loss, 'perplexity': perplexity, 'teacher_loss': teacher_loss, 'teacher_perplexity': teacher_perplexity}
        else:
            return {'loss': loss, 'perplexity': perplexity}
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # 在每个训练批次结束时清空缓存
        try:
            get_accelerator().empty_cache()
        except AttributeError:
            # 如果get_accelerator()不可用,尝试使用torch.cuda
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"无法清空缓存: {e}")
    def training_step(self, batch, batch_idx):
        args = self.args
        teacher_model = self.teacher_model
        tokenizer = self.tokenizer
        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = torch.ne(input_ids, tokenizer.eos_token_id).to(input_ids.device)
        if not args.is_sft:
            if args.teacher_client_mode:
                b,t = input_ids.shape
                # logging.info(input_ids.dtype)
                # logging.info(input_ids.shape)
                logging.info(f'rank {args.rank} is sending input_ids to server, shape is {input_ids.shape}')
                self.comm.send(input_ids.data_ptr(), input_ids.size(0)*input_ids.size(1), nccl.NCCL_INT64, args.server_rank, self.stream.ptr)
                self.stream.synchronize()
                logging.info(f'rank {args.rank} is receiving teacher_logits from server')
                self.comm.recv(self.recv_buffer.data.ptr, self.recv_buffer.size, nccl.NCCL_FLOAT, args.server_rank, self.stream.ptr)
                self.stream.synchronize()
                teacher_logits = torch.as_tensor(self.recv_buffer, device=input_ids.device, dtype=torch.float32)
                logging.info(f'rank {args.rank} is receiving teacher_logits from server, shape is {teacher_logits.shape}')
                if args.is_hidden_align:
                    logging.info(f'rank {args.rank} is receiving teacher_hidden_states from server')
                    self.comm.recv(self.teacher_hidden_states_buffer.data.ptr, self.teacher_hidden_states_buffer.size, nccl.NCCL_FLOAT, args.server_rank, self.stream.ptr)
                    self.stream.synchronize()
                    logging.info(f'rank {args.rank} is receiving teacher_hidden_states from server, shape is {self.teacher_hidden_states_buffer.shape}')
                    teacher_hidden_states = torch.as_tensor(self.teacher_hidden_states_buffer, device=input_ids.device, dtype=torch.float32)#(b*layers,t,hidden_size)
                    logging.info(f'got teacher hidden states shape is {teacher_hidden_states.shape}, rank is {args.rank}')
            else:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False,output_hidden_states=args.is_hidden_align)
                teacher_logits = teacher_outputs.logits
                if args.is_hidden_align:
                    teacher_hidden_states = teacher_outputs.hidden_states    
                    teacher_hidden_states = torch.cat(teacher_hidden_states[1:], dim=0)#(b*layers,t,hidden_size)
            
            # teacher_logits = teacher_logits.detach()
            student_outputs = self.forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False,output_hidden_states=args.is_hidden_align)
            if not args.is_hidden_align:
                # 假设 labels 是输入的真实标签
                targets = F.softmax(teacher_logits, dim=-1)
                student_logits = student_outputs.logits
                student_cross_entropy_loss = student_outputs.loss

                # 创建一个掩码，标记不是 -100 的位置
                mask = (labels != -100).float()

                # 计算 log_softmax，但只在非 -100 的位置
                log_probs_student = F.log_softmax(student_logits, dim=-1) * mask.unsqueeze(-1)
                probs_teacher = targets * mask.unsqueeze(-1)
                # del targets
                # del student_logits
                # torch.cuda.empty_cache()
                # print(f'log_probs_student shape is {log_probs_student.shape}, probs_teacher shape is {probs_teacher.shape}')
                # 计算 KL 散度，忽略 -100 的位置
                kl_div = F.kl_div(log_probs_student, probs_teacher, reduction='none')
                kl_div = kl_div.sum(dim=-1)  # 在词汇表维度上求和

                # 计算非 -100 标记的数量
                num_valid_elements = mask.sum()

                # 计算平均 KL 散度，只考虑非 -100 的位置
                kl_loss = kl_div.sum() / num_valid_elements
                # kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), targets, reduction='batchmean')
                loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
                self.log('train_loss', loss)
                returned_loss = {}
                returned_loss['loss'] = loss
                returned_loss['kl_loss'] = kl_loss
                returned_loss['student_cross_entropy_loss'] = student_cross_entropy_loss
                return returned_loss
            else:
                logging.info(f'rank {args.rank} training with hidden states alignment')
                mask = torch.ne(labels, -100).to(labels.device)
                mask = mask.unsqueeze(1).unsqueeze(3)#(b,1,t,1)
                hidden_states = torch.cat(student_outputs.hidden_states[1:], dim=0)#(b*layers,t,hidden_size)
                hidden_states = hidden_states * mask
                teacher_hidden_states = teacher_hidden_states * mask
                logging.info(f'students hidden shape is {hidden_states.shape}, rank is {args.rank}')
                logging.info(f'teachers hidden shape is {teacher_hidden_states.shape}, rank is {args.rank}')
                loss = F.mse_loss(hidden_states, teacher_hidden_states.to(hidden_states.dtype))
                self.log('train_loss', loss)
                logging.info(f'rank {args.rank} finished training with hidden states alignment, loss is {loss}')
                return loss
        else:
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
            loss = outputs.loss
            self.log('train_loss', loss)
            return loss
            
