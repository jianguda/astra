from typing import Dict, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..hparams import AlphaEditHyperParams
from . import nethook


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: AlphaEditHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(lm_w.shape[1])

    #print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(data["answer"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]  
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    cur_input_ids = input_tok['input_ids'] 
    all_delta = []
    all_target = []
    all_idxs = []
    # ans_sen = nltk.tokenize.sent_tokenize(data["answer"], language='english')
    # cur_sen = ""
    # for i,sen in enumerate(ans_sen):
    #     if i > 0  and sen[0]!=" ":
    #         sen = " " + sen
    #     cur_sen = cur_sen + sen
    #     current_target_ids = tok(cur_sen, return_tensors="pt").to("cuda")["input_ids"][0]
    #     if current_target_ids[0] == tok.bos_token_id or current_target_ids[0] == tok.unk_token_id:
    #         current_target_ids = current_target_ids[1:]
    #     # if len(current_target_ids) < hparams.window_size:
    #     #     continue
    #     input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[:-1], dim=0)], dim=1)
    #     cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids, dim=0)], dim=1)
        

    start = 0
    while start < len(target_ids):
        end = start + hparams.window_size
        if end > len(target_ids):
            end = len(target_ids)
        current_target_ids = target_ids[start:end]
        if start > 0:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:], dim=0)], dim=1)
        else:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids, dim=0)], dim=1)
        
        start += hparams.window_size - hparams.overlap
        
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            1, len(input_ids[0])
        )
   
        ex_len = len(input_ids[0])
    
        rewriting_targets[0, ex_len - len(current_target_ids) : ex_len] = current_target_ids


        lookup_idxs = [ex_len - len(current_target_ids)]
    
        loss_layer = max(hparams.v_loss_layer, layer)
    
        # the optimized residual is kept in float32, whatever the precision of the model
        delta = torch.zeros((lm_w.shape[0],), requires_grad=True, device="cuda")
        target_init = None
    
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init  

            if cur_layer == hparams.layer_module_tmp.format(layer):
                
                if isinstance(cur_out, tuple):
                    out_tensor = cur_out[0]
                    is_tuple = True
                else:
                    out_tensor = cur_out
                    is_tuple = False
                
                if out_tensor.dim() == 2:
                    out_tensor = out_tensor.unsqueeze(0)
                
                if target_init is None:
                    target_init = out_tensor[0, lookup_idxs[0]].detach().clone()

                for idxs_pre, delta_pre in all_delta:
                    for i, idx in enumerate(idxs_pre):
                        if len(idxs_pre) != out_tensor.shape[1]:
                            out_tensor[0, idx, :] += delta_pre
                        else:
                            out_tensor[0, i, :] += delta_pre
                
                for i, idx in enumerate(lookup_idxs):
                    if len(lookup_idxs) != out_tensor.shape[1]:
                        out_tensor[0, idx, :] += delta
                    else:
                        out_tensor[0, i, :] += delta
                
                if is_tuple:
                    if len(cur_out) > 1:
                        cur_out = (out_tensor,) + cur_out[1:]
                    else:
                        cur_out = (out_tensor,)
                else:
                    cur_out = out_tensor

            return cur_out

        # Optimizer
        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)  

        # Execute optimization
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids).logits
                

            # Compute loss on rewriting targets

            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            
            if output.dim() == 2:
                output = output.unsqueeze(0)
            
            if output.shape[0] != input_ids.shape[0]:
                if output.shape[1] == input_ids.shape[0]:
                    output = output.transpose(0, 1)
            
            if output.shape[1] != rewriting_targets.shape[1]:
                if output.shape[0] == rewriting_targets.shape[1]:
                    output = output.transpose(0, 1)
            
            full_repr = output

            # logits in the model's precision, the loss in float32
            log_probs = torch.log_softmax((ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device)).float(), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / current_target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + weight_decay.to(nll_loss.device)
            print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}  + {np.round(weight_decay.item(), 3)} "
            # f"avg prob of [{cur_sen}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 1e-2:
               break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()
        # cur_sen = ""
        target = target_init.float() + delta
        all_delta.append((lookup_idxs ,delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        print(
        f"Iteration {len(all_delta)}: Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return all_idxs, all_target




