import sys
import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
from decision_transformer.models.utils import cross_entropy, encode_return

class SequenceTrainer(Trainer):
    def train_step(self):
        if self.trajectory_example:
            (
                states,
                actions,
                rewards,
                dones,
                rtg,
                timesteps,
                attention_mask,
            ) = self.get_batch(self.batch_size)
            
            (
                e_states,
                e_actions,
                e_rewards,
                e_dones,
                e_rtg,
                e_timesteps,
                e_attention_mask,
            ) = self.get_batch(self.batch_size)

            action_target = torch.clone(actions)

            observation_preds, action_preds, _, _ = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                e_states,
                e_actions,
                e_rewards,
                e_rtg,
                e_timesteps,
                attention_mask=attention_mask,
            )
        elif self.args["visualize_attn"] and not self.args["eval_only"]:
            self.model.eval()
            (
                states,
                actions,
                rewards,
                dones,
                rtg,
                timesteps,
                attention_mask,
            ) = self.get_test_batch(1)

            action_target = torch.clone(actions)
            with torch.no_grad():
                observation_preds, action_preds, rtg_preds, heatmap_list = self.model.forward(
                    states,
                    actions,
                    rewards,
                    rtg[:, :-1],
                    timesteps,
                    attention_mask=attention_mask,
                )

            print(f"{heatmap_list[0].shape=}") # [N, 12, seq_len, seq_len]
            all_layer_list = []

            n_head = heatmap_list[0][0].shape[1]
            for layer_id in range(1): # n_layer

                ret_list = []
                for head_id in range(12): # iterate through heads

                    last_row_list = []
                    episode_len = heatmap_list[0].shape[0]
                    for time_step in range(episode_len): # iterate through episode
                        #print(f"{hm_all[0].shape=}") # bs, n_head, seq, seq
                        hm = heatmap_list[layer_id][time_step][head_id]

                        if hm.shape[-1] == 60:
                            last_row = hm[-2, :] # get the last-but-one line for action prediction

                            #cum_last_row = np.cumsum(last_row.detach().cpu().numpy()[::-1]) # get rid of the last action, which is a zero tensor
                            #step_wise_hm = cum_last_row[2::3] # step_wise_hm should be [20]
                            last_row = last_row.detach().cpu().numpy()[::-1]
                            #step_wise_hm = cum_last_row[1:]
                            step_wise_hm = last_row[1:]

                            last_row_list.append(step_wise_hm)
                    if (len(last_row_list) > 0):
                        final_last_row = sum(last_row_list) / len(last_row_list)
                        ret_list.append(final_last_row)
                        #print(f"{final_last_row=}")
                    else:
                        print(f"{len(heatmap_list)=}")

                all_layer_list.append(ret_list)
            np.set_printoptions(threshold=sys.maxsize)
            print(repr(np.array(all_layer_list)))
            raise NotImplementedError

        else:
            (
                states,
                actions,
                rewards,
                dones,
                rtg,
                timesteps,
                attention_mask,
            ) = self.get_batch(self.batch_size)

            action_target = torch.clone(actions)

            observation_preds, action_preds, rtg_preds, _ = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                attention_mask=attention_mask,
            )

        self.step += 1
        act_dim = action_preds.shape[2]
        #print(f"{attention_mask.shape=}") # [64, 20]

        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]

        action_loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )
        loss = action_loss

        if rtg_preds != None:
            rtg_target = (
                encode_return(
                    self.args["env"],
                    rtg[:, :-1],
                    num_bin=self.args["num_bins"],
                    rtg_scale=self.args["rtg_scale"],
                )
                .float()
                .reshape(-1, 1)[
                    attention_mask.view(-1,) > 0
                ]
            )
            rtg_preds = rtg_preds.reshape(-1, self.args["num_bins"])[
                attention_mask.view(-1,) > 0
            ]

            # print(f"{rtg_preds.shape=}, {rtg_target.shape=}")
            # rtg_preds.shape=torch.Size([1247, 60]), rtg_target.shape=torch.Size([1247, 1])
            rtg_loss = cross_entropy(rtg_preds, rtg_target, self.args["num_bins"])

            with torch.no_grad():
                self.diagnostics["training/rtg_ce_loss"] = (
                    rtg_loss.detach().cpu().item()
                )
            loss += self.args["rtg_weight"] * rtg_loss

        batch = next(self.train_nlp_dataset)

        if self.args["co_training"]:
            lm_out = self.model.transformer_model(**batch)
            lm_loss = lm_out.loss
            loss += self.args["co_lambda"] * lm_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                action_loss.detach().cpu().item()
                #torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()#, lm_loss.detach().cpu().item()