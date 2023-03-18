'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

MutexMatch with threshold - mutexmatch with threshold class
Here named as just MutexMatch
code adapted from https://github.com/NJUyued/MutexMatch4SSL 
'''

import os
import math
import torch
import contextlib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.metrics import f1_score

from models.nets.net import *
from train_utils import ce_loss
from torch.cuda.amp import autocast, GradScaler
from .mutexmatch_utils import consistency_loss, Get_Scalar 

# The peformance metric definition  
def EXPR_metric(x, y): 

    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')

    return f1

class TotalNet(nn.Module):
    def __init__(self, net_builder, num_classes, load=None):
        super(TotalNet, self).__init__()

        if load == './models/nets/resnet18_msceleb.pth':
            self.feature_extractor = net_builder(num_classes=num_classes, load= load)
            checkpoint = torch.load('./models/nets/resnet18_msceleb.pth')
            self.feature_extractor.backbone.load_state_dict(checkpoint['state_dict'], strict=True)
            self.feature_extractor.backbone = nn.Sequential(*list(self.feature_extractor.backbone.children())[:-1])
        else:
            self.feature_extractor = net_builder(num_classes=num_classes)  

        classifier_output_dim = num_classes
        self.classifier_reverse = ReverseCLS(self.feature_extractor.output_num(), classifier_output_dim)
        
    def forward(self, x):
        f = self.feature_extractor(x)
        return f

class MutexMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, k=0, t_fn=None, p_fn=None, it=0, num_eval_iter=1000,\
                 tb_log=None, logger=None, data=None, load=None):
     
        super(MutexMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m
        self.p_cutoff = p_cutoff
        print(f' In Mutex Class self.p_cutoff : {self.p_cutoff}')

        if load:
            self.train_model = TotalNet(net_builder, num_classes, load)       
            self.eval_model = TotalNet(net_builder, num_classes, load) 
        else:
            self.train_model = TotalNet(net_builder, num_classes)       
            self.eval_model = TotalNet(net_builder, num_classes) 
            self.p_fn = Get_Scalar(p_cutoff) # confidence cutoff function

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) # temperature params function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        self.k = k
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')   
        for key in  self.loader_dict.keys():
            print(f"{key} -> len: {len(self.loader_dict[key])}")
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def train(self, args, logger=None):
        """
        Train function of MutexMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        print("In train")
        feature_extractor = self.train_model.module.feature_extractor.train(True) if hasattr(self.train_model, 'module') else self.train_model.feature_extractor.train(True)
        cls_reverse = self.train_model.module.classifier_reverse.train(True) if hasattr(self.train_model, 'module') else self.train_model.classifier_reverse.train(True)
        feature_extractor.cuda(args.gpu)
        cls_reverse.cuda(args.gpu)
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_f1, best_it = 0.0, 0
        p_cutoff = self.p_cutoff
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        p_bar = tqdm(range(len(self.loader_dict['train_lb']))) 

        for (x_lb, y_lb), data in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            y_lb = y_lb.long()
            if args.dataset == 'aff':
                # print(data)
                x_ulb_w = data[0][0]
                x_ulb_s = data[0][1]
            else:
                assert Exception("Not Implemented Error")

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            # @V[5 Feb 23]
            # RuntimeError: CUDA error: unspecified launch failure 
            y_lb = y_lb.cuda(args.gpu)
            
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            
            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, feature = feature_extractor(inputs, ood_test=True) 
                # logits, (feature_extractor O/P->features{for all labeled, unlabeled(w,s)})
                
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2) # divide into two appropriate groups
                # Waug logits, Saug logits

                ##
                # I/P,   fc O/P,        , Softmax O/P               {for all labeled, unlabeled(w,s)}
                feature, logits_reverse, predict_prob = cls_reverse(feature)
                logits_x_ulb_w_reverse, logits_x_ulb_s_reverse = logits_reverse[num_lb:].chunk(2)

                # I/P,   fc O/P,        , Softmax O/P (with detach-> no computation graph associated with these calculations)
                #                                                                   #  {for all labeled, unlabeled(w,s)}
                feature_separate, logits_reverse_separate, predict_prob_separate = cls_reverse(feature.detach())

                #                         #  {for all labeled, unlabeled(w,s)}
                complementary_label = torch.softmax(logits.detach(), dim=-1)        
                min_probs_reverse, min_idx_reverse = torch.min(complementary_label, dim=-1)
                #                  class with minimum index {for all labeled, unlabeled(w,s)}

                
                # hyper-params for update
                T = self.t_fn(self.it)
                del logits
              
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean') # Lsup
                reverse_loss = ce_loss(logits_reverse_separate, min_idx_reverse, reduction='mean')
            
                assert self.k<=self.num_classes and self.k>=0
                unsup_loss, masked_reverse_loss, mask = consistency_loss(
                                              logits_x_ulb_w_reverse,
                                              logits_x_ulb_s_reverse,
                                              logits_x_ulb_w, 
                                              logits_x_ulb_s,  
                                              self.k,                                                                  
                                              'ce', T, p_cutoff,
                                               use_hard_labels=args.hard_label)
                total_loss = sup_loss + self.lambda_u * unsup_loss + reverse_loss + self.lambda_u * masked_reverse_loss        
                    
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward() 
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach() 
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/reverse_loss'] = reverse_loss.detach()
            tb_dict['train/masked_reverse_loss'] = masked_reverse_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            if self.it % self.num_eval_iter == 0: # @v[17 Feb 23]
                _, _, logits_x_lb_new, targets_x_lb_new = self.evaluate_fast(args=args, eval_loader=self.loader_dict['train_l_eval'], p_cutoff=self.p_cutoff)
                p_cutoff = self.adaptive_threshold_generate(logits_x_lb_new, targets_x_lb_new, p_cutoff, self.it, args)
                print("Generated threshold : ", p_cutoff)
                eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],p_cutoff=p_cutoff) # @v
                tb_dict.update(eval_dict) # @v
                
                save_path = os.path.join(args.save_dir, args.save_name) # @v
                
                if tb_dict['eval/f1'] > best_eval_f1:
                    best_eval_f1 = tb_dict['eval/f1']
                    best_it = self.it# @v
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_F1: {best_eval_f1}, at {best_it} iters") # @v
            
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    acc = "".join('%.2f' % (best_eval_f1.item()*100)) # @V
                    self.save_model(f'model-f1-{acc}.pth', save_path) # @V
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                    

            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000

            p_bar.set_description("LR: {lr:.3f}. Loss: {loss:.3f}. Lsup: {lsup:.3f}. Lsep: {lsep:.3f}. Lp: {lp:.3f}. Ln: {ln:.3f}".format(
                lr=self.scheduler.get_last_lr()[0], 
                loss=total_loss.detach().cpu().numpy(),
                lsup=sup_loss.detach().cpu().numpy(),
                lsep=reverse_loss.detach().cpu().numpy(),
                lp=unsup_loss.detach().cpu().numpy(), 
                ln=masked_reverse_loss.detach().cpu().numpy()))
            p_bar.update()
        p_bar.close()
        
        eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],ulb_loader=self.loader_dict['eval_ulb'],p_cutoff=p_cutoff)
        eval_dict.update({'eval/best_f1': best_eval_f1, 'eval/best_it': best_it})
        return eval_dict


    def adaptive_threshold_generate(self,outputs, targets, threshold, epoch, args=None):

        outputs_l = outputs[1:, :]
        targets_l = targets[1:]
        probs = torch.softmax(outputs_l, dim=1)
        max_probs, max_idx = torch.max(probs, dim=1)
        eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]
        probs_new = max_probs[eq_idx]
        targets_new = targets_l[eq_idx]
        for i in range(self.num_classes):
            idx = np.where(targets_new.cpu() == i)[0]
            if idx.shape[0] != 0:
                cur_t = probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch))
                threshold[i] = cur_t if cur_t >= args.p_cutoff else args.p_cutoff
            else:
                threshold[i] = args.p_cutoff

        return threshold

    @torch.no_grad()
    def evaluate_fast(self, eval_loader=None, args=None, p_cutoff=None):

        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        feature_extractor = self.eval_model.module.feature_extractor if hasattr(self.eval_model, 'module') else self.eval_model.feature_extractor

        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0    

        p_bar = tqdm(range(len(eval_loader)))  

        outputs_new = torch.ones(1, args.num_classes).cuda()
        targets_new = torch.ones(1).long().cuda()  

        for idx, (x, y) in enumerate(eval_loader):
            y = y.long()
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)

            num_batch = x.shape[0]
            total_num += num_batch

            logits, feature = feature_extractor(x, ood_test=True)           
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)

            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(max_idx == y)        
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()

            outputs_new = torch.cat((outputs_new, logits), dim=0)
            targets_new = torch.cat((targets_new, y), dim=0)
                
            p_bar.set_description(" Loss: {loss:.4f}. Acc: {acc:.4f}".format( loss=loss, acc=acc))
            p_bar.update()
        p_bar.close()  

        exp_f1 = EXPR_metric(outputs_new.cpu().numpy() , targets_new.cpu().numpy())
        print('\nexp performance on train : {:4f}\n'.format(exp_f1)) 
            
        
        return total_loss/total_num, total_acc/total_num, outputs_new, targets_new

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None , lb_loader=None ,ulb_loader=None,p_cutoff=None):
        
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        feature_extractor = self.eval_model.module.feature_extractor if hasattr(self.eval_model, 'module') else self.eval_model.feature_extractor

        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0      

        bar = tqdm(range(len(eval_loader)))  

        outputs_new = torch.ones(1, 8).cuda()
        targets_new = torch.ones(1).long().cuda()

        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)

            num_batch = x.shape[0]
            total_num += num_batch

            logits, feature = feature_extractor(x, ood_test=True)           
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)

            loss = F.cross_entropy(logits, y, reduction='mean')

            outputs_new = torch.cat((outputs_new, logits), dim=0)
            targets_new = torch.cat((targets_new, y), dim=0)

            acc = torch.sum(max_idx == y)        
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()       

            bar.set_description("Eval Loader: loss: {loss:.3f}. acc: {acc:.3f}. totalloss: {total_loss:.3f}. total_acc : {total_acc:.3f}".format(
                loss=loss.detach().cpu().numpy(),
                acc=acc.detach().cpu().numpy(),
                total_loss=total_loss.detach().cpu().numpy(),
                total_acc=total_acc.detach().cpu().numpy(),
                )) 
            bar.update()
        bar.close()     

        exp_f1 = EXPR_metric(outputs_new.cpu().numpy() , targets_new.cpu().numpy())
        print('\nexp performance : {:4f}\n'.format(exp_f1)) 

        return {'eval/loss': total_loss/total_num, 'eval/f1': exp_f1,'eval/top-1-acc': total_acc/total_num}
    
    
    def mask_generate_greaternequal(self, max_probs, max_idx, batchsize, threshold):
        mask = torch.zeros(batchsize)
        for i in range(self.num_classes):
            idx = np.where(max_idx.cpu() == i)[0]
            m = max_probs[idx].ge(threshold[i]).float()
            for k in range(len(idx)):
                mask[idx[k]]+=m[k].cpu()
        return mask.cuda()


    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")

    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key], strict=False)
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key], strict=False)
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")
    
    def load_model_from_chkpth(self, load_path):
        checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        train_model_state_dict = train_model.state_dict()
        eval_model_state_dict = eval_model.state_dict()
        dont_load = ['feature_extractor.module.classifier.bias',
                     'feature_extractor.module.classifier.weight',
                     'classifier_reverse.module.fc.weight',
                     'classifier_reverse.module.fc.bias',
                     'classifier_reverse.module.main.0.weight',
                     'classifier_reverse.module.main.0.bias']

        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    for param in checkpoint[key]:
                        if param in dont_load:
                            continue
                        else:
                            train_model_state_dict[param] = checkpoint[key][param]
                    train_model.load_state_dict(train_model_state_dict, strict=False)
                elif 'eval_model' in key:
                    for param in checkpoint[key]:
                        if param in dont_load:
                            continue
                        else:
                            eval_model_state_dict[param] = checkpoint[key][param]
                    eval_model.load_state_dict(eval_model_state_dict, strict=False)
                else:
                    continue
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass
