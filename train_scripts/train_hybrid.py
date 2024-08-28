import sys
import os
def setup_env():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)
    rwkv_path = os.path.join(parent_dir, 'rwkv')
    sys.path.append(rwkv_path)
    rwkv_llama_path = os.path.join(parent_dir, 'rwkv_llama')
    sys.path.append(rwkv_llama_path)
    print(f'add path: {rwkv_path} to sys.path')
    print(f'add path: {rwkv_llama_path} to sys.path')
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.1'
    os.environ['RWKV_JIT_ON'] = '0'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = 'fla'
    os.environ["RWKV_TRAIN_TYPE"] = ''
setup_env()
import argparse
from argparse import Namespace
def create_arg_parser():
    parser = argparse.ArgumentParser(description='MLM trainer')
    parser.add_argument('--config_file', type=str,default='configs/test_hybrid.yaml', help='training config file')
    parser.add_argument('--train_data', type=str,help='parquet dicrectory containing the training data')
    parser.add_argument('--c4_data', type=str,help='c4 data directory')
    parser.add_argument('--languages', type=str,nargs='+',default=['en','zh'],help='languages to train the model')
    parser.add_argument('--output_dir', type=str, default='/data/rwkv/tmp',help='directory to save the trained model')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--max_seq_length', type=int, default=512, help='maximum sequence length to train the model')
    parser.add_argument('--num_devices', type=int, default = 1,help='number of devices to train the model')
    
    
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate in the model')
    parser.add_argument('--grad_cp', type=int, default=0, help='gradient checkpoint in the model')
    parser.add_argument('--save_per_batches', type=int, default=10000, help='number of batches to save the model')
    parser.add_argument('--my_exit', type=int, default=300, help='exit condition in the model')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay in the model')
    parser.add_argument('--lr_init', type=float, default=6e-4, help='initial learning rate in the model')
    parser.add_argument('--lr_final', type=float, default=1e-5, help='final learning rate in the model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 parameter in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2 parameter in the Adam optimizer')
    parser.add_argument('--layerwise_lr', type=float, nargs='+', default=1, help='layerwise learning rate in the model')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon parameter in the Adam optimizer')
    parser.add_argument('--warmup_steps', type=int, default=50, help='warmup steps in the model')
    parser.add_argument('--epoch_begin', type=int, default=0, help='beginning epoch for the training')
    parser.add_argument('--epoch_count', type=int, default=150, help='total number of epochs for the training')
    parser.add_argument('--epoch_save', type=int, default=1, help='number of epochs after which the model is saved')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='number of epochs after which the validation is checked')
    parser.add_argument('--val_check_interval', type=int, default=5000, help='number of epochs after which the validation is checked')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='number of validation steps for sanity check at the beginning of training')
    parser.add_argument('--log_every_n_steps', type=int, default=5000, help='number of steps after which the training progress will be logged')
    parser.add_argument('--enable_checkpointing', type=bool, default=False, help='flag to enable checkpointing')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='number of batches to accumulate before performing a backward/update pass')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='maximum gradient norm')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--micro_bsz', type=int,default=2, help='micro batch size for training')
    parser.add_argument('--real_bsz', type=int, help='real batch size for training')
    parser.add_argument('--my_pile_stage', type=int, default=0, help='pile stage in the model')
    parser.add_argument('--my_pile_edecay', type=float, default=0, help='pile exponential decay in the model')
    parser.add_argument('--weight_decay_final', type=float, default=-1, help='final weight decay in the model')
    parser.add_argument('--proj_dir', type=str, help='project directory to save the model and logs')
    parser.add_argument('--eval_every_steps', type=int, default=100, help='number of steps after which the model is evaluated')
    parser.add_argument('--wandb', type=str, default='hybrid_trainer', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='hybrid_trainer_a800', help='run name for wandb logging')
    parser.add_argument('--strategy', type=str, default='deepspeed_stage_2_offload', help='strategy for distributed training')
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument('--my_qa_mask', type=int, default=0)
    parser.add_argument('--optim',type=str,default='adam',help='optimizer')
    parser.add_argument('--train_type', type=str, default='', help='train type')
    parser.add_argument('--skip_steps',type=int,default=0,help='skip steps in the peft checkpoint')

    parser.add_argument('--ckpt_file', type=str, default=None, help='checkpoint file')
    return parser

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    print(args)
    import yaml
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    import torch
    dtype = torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformer_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'],
                                                            torch_dtype=dtype, device_map={'':'cpu'})
    print(transformer_model.config)
    tokenizer = AutoTokenizer.from_pretrained(config['Llama']['model_id'])
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.eos_token_id)


    args.my_pos_emb = 0
    args.head_size_a = 64
    args.head_size_divisor = 8
    args.ctx_len = 4096
    args.n_layer = transformer_model.config.num_hidden_layers
    args.n_embd = transformer_model.config.hidden_size
    args.dim_att = transformer_model.config.hidden_size
    args.dim_ffn = transformer_model.config.intermediate_size
    args.pre_ffn = 0
    args.head_qk = 0
    args.tiny_att_dim = 0
    args.tiny_att_layer = -999
    args.vocab_size = transformer_model.config.vocab_size
    args.layers = config['RWKV']['layers']
    args.pad_id = tokenizer.eos_token_id
    args.betas = (args.beta1, args.beta2)
    args.kl_weight = config['kl_weight']
    args.ce_weight = config['ce_weight']
    args.model_file = config['model_file']
    args.real_bsz = args.micro_bsz * args.accumulate_grad_batches*args.num_devices*args.num_nodes
    args.teacher_client_mode = config['teach_mode']['is_client']
    args.nccl_file = config['teach_mode']['nccl_file']
    args.num_groups = config['teach_mode']['num_groups']
    args.is_hidden_align = config['teach_mode']['is_hidden_align']
    
    assert args.num_devices % args.num_groups == 0
    import time
    args.my_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    from hybrid_model import HybridModel


    if not args.teacher_client_mode:
        teacher_model = AutoModelForCausalLM.from_pretrained(config['Llama']['model_id'], torch_dtype=dtype)
    else:
        teacher_model = None
        

    model = HybridModel(transformer_model,args,teacher_model,tokenizer)

    for name, param in model.named_parameters():
        if not 'block.' in name:
            param.requires_grad = False
        print(name, param.shape, param.requires_grad)
    import datasets
    if args.train_data is not None:
        print(f'load train data from {args.train_data}')
        from datasets import load_from_disk
        ds = load_from_disk(args.train_data)
        print(ds)
        def data_collator(features):
            input_ids = []
            labels = []
            for f in features:
                input_ids.append(f['input_ids'])
                labels.append(f['labels'])
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            return {'input_ids': input_ids, 'labels': labels}  
        train_dataloader = torch.utils.data.DataLoader(ds, 
                                                batch_size=args.micro_bsz,
                                                shuffle=True, 
                                                num_workers=4, 
                                                pin_memory=True, 
                                                drop_last=True,
                                                collate_fn=data_collator)
        val_dataloader = None
    elif args.c4_data is not None:
        print(f'load c4 data from {args.c4_data}')
        from data.c4_datasets import load_and_interleave_c4,data_collator
        train_ds = load_and_interleave_c4(args.c4_data, args.languages, split='train')
        # train_ds = train_ds[:100000]
        val_ds = load_and_interleave_c4(args.c4_data, args.languages, split='validation')  
        data_collator = data_collator(tokenizer, max_seq_length=args.max_seq_length)
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.micro_bsz, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=data_collator)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.micro_bsz, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, collate_fn=data_collator)
        

    args.epoch_steps = len(train_dataloader)//(args.num_devices*args.num_nodes)
    from pytorch_lightning import Trainer
    precision = 'bf16'
    from Callbacks import TrainerCallback
    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          filename='{epoch}-{step}-{train_loss:.4f}',
                                          save_top_k=1,
                                          every_n_train_steps=5,
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True)
    trainer = Trainer(accelerator="auto",
                      strategy=args.strategy,
                      devices=args.num_devices,
                      num_nodes=args.num_nodes,
                      precision=precision,
                      logger=False,
                      callbacks=[TrainerCallback(args),checkpoint_callback],
                      max_epochs=args.max_epochs,
                      check_val_every_n_epoch=None,
                      val_check_interval=args.val_check_interval,
                      num_sanity_val_steps=args.num_sanity_val_steps,
                      log_every_n_steps=args.log_every_n_steps,
                      enable_checkpointing=args.enable_checkpointing,
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      gradient_clip_val=args.gradient_clip_val)
    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
    
    print(model)
    if args.ckpt_file is not None:
        dict_set = torch.load(args.ckpt_file)
        info = model.load_state_dict(dict_set, strict=False)
        print(f'load model from {args.ckpt_file}, info is {info}')
        del dict_set
    model.train()
    print("Current device rank: ", trainer.global_rank)
    print("Total number of devices: ", trainer.world_size)
    args.world_size = trainer.world_size
    args.rank = trainer.global_rank
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model, 
                train_dataloader,
                val_dataloader)