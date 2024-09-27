import math
import time
import pytorch_lightning as pl
import torch
import os

def save_trainable_parameters(model, trainable_dir_output, model_filename):
    print('Do nothing')
    # print(f"save trainable parameters to {trainable_dir_output} pretrained from {model_filename}")
    # # 创建保存目录
    # os.makedirs(trainable_dir_output, exist_ok=True)
    
    # # 获取可训练的参数
    # trainable_params = []
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         trainable_params.append(param)
    
    # # 判断是否有可训练的参数
    # if len(trainable_params) == 0:
    #     print("没有可训练的参数")
    #     return

    # # 保存可训练的参数
    # save_filename = os.path.basename(model_filename) + '.pth'
    # save_path = os.path.join(trainable_dir_output, save_filename)
    # state_dict = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
    # torch.save(state_dict, save_path)
    # print(f"save trainable parameters to {save_path}")
class TrainerCallback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.eval_loss = []
        self.args = args
        self.wandb_init = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps
        if args.lr_final == args.lr_init or args.epoch_count == 0:
            lr = args.lr_init
        else:
            decay_step = real_step - args.my_pile_edecay * args.epoch_steps
            decay_total = (args.epoch_count - args.my_pile_edecay) * args.epoch_steps
            progress = (decay_step - w_step + 1) / (decay_total - w_step)
            progress = min(1, max(0, progress))

            if args.lr_final == 0 or args.lr_init == 0:  # linear decay
                lr = args.lr_init + (args.lr_final - args.lr_init) * progress
            else:  # exp decay
                lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
            # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay


        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr
                
        trainer.my_lr = lr
        trainer.my_wd = wd_now
                
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.is_global_zero:
            if  trainer.global_step == 0: # logging
                os.makedirs(args.output_dir, exist_ok=True)
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.output_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0 and self.wandb_init != True:
                    print("Login to wandb...")
                    import wandb
                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + args.my_timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb
                    self.wandb_init = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.max_seq_length * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero and batch_idx > args.skip_steps:  # logging   
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_time_ns = t_now
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                # returned_loss['bow_loss'] = bow_loss
                # returned_loss['enc_loss'] = enc_loss
                # returned_loss['decoder_loss'] = decoder_loss
                lll = {"loss": trainer.my_loss,  
                        "Gtokens": real_step * token_per_step / 1e9}
                if 'kl_loss' in outputs:
                    lll["kl_loss"] = outputs['kl_loss']
                if 'teacher_cross_entropy_loss' in outputs:
                    lll["teacher_cross_entropy_loss"] = outputs['teacher_cross_entropy_loss']
                if 'student_cross_entropy_loss' in outputs:
                    lll["student_cross_entropy_loss"] = outputs['student_cross_entropy_loss']
                if 'decoder_loss' in outputs:
                    lll["decoder_loss"] = outputs['decoder_loss']
                if 'key_match_loss' in outputs:
                    lll["key_match_loss"] = outputs['key_match_loss']
                if 'value_match_loss' in outputs:
                    lll["value_match_loss"] = outputs['value_match_loss']
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))
            
            if real_step % args.log_every_n_steps == 0 and real_step > 0:
                print(f'saving trainable to {args.output_dir}')
                print(f"{real_step} {trainer.my_loss:.6f} {math.exp(trainer.my_loss):.4f}  {trainer.current_epoch}, now saving...")
                output_dir = f"{args.output_dir}/epoch_{trainer.current_epoch}_step_{real_step}"
                save_trainable_parameters(pl_module, output_dir, args.model_file)
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        # dataset.global_rank = trainer.global_rank
        # dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        # dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f}  {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
                exit(0)
            # output_dir = f"{args.output_dir}/epoch_{trainer.current_epoch}"
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # save_trainable_parameters(pl_module, output_dir, args.model_file)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.is_global_zero:
            if pl.__version__[0]=='2':
                self.eval_loss.append(outputs["loss"].item())
            else:
                self.eval_loss.append(trainer.my_loss_all.float().mean().item())
        
         # 在每个训练批次结束时清空缓存
        try:
            from deepspeed.accelerator import get_accelerator
            get_accelerator().empty_cache()
        except AttributeError:
            # 如果get_accelerator()不可用,尝试使用torch.cuda
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"无法清空缓存: {e}")
    def on_validation_start(self,trainer,pl_module) -> None:
        self.eval_loss = []

    def on_validation_end(self,trainer,pl_module) -> None:
        if trainer.is_global_zero:
            my_val_loss = sum(self.eval_loss) / len(self.eval_loss)
            if len(self.args.wandb) > 0:
                trainer.my_wandb.log({"val_loss": my_val_loss}, step=int(trainer.global_step))