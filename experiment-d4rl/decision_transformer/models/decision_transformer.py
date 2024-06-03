import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import transformers

import sys
import os
# use ../../decision_transformer as decision_transformer when run as main
if __name__=="__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.model import TrajectoryModel
from transformers.models.gpt2 import GPT2Tokenizer
from decision_transformer.models.trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Config_LoRA
from decision_transformer.models.image_gpt2_LoRA import GPT2LMHeadModel_LoRA as iGPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_llama2 import LlamaModel, LlamaForCausalLM

from transformers import LlamaConfig

from decision_transformer.models.utils import ResidualBlock, MLPBlock

class StateAbstractionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(StateAbstractionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    @property
    def transformer(self):
        if self.args["pretrained_lm"] is not None:
            if "gpt" in self.args["pretrained_lm"]:
                return self.transformer_model.transformer
            else:
                return self.transformer_model

    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.args = args
        self.hidden_size = hidden_size

        self.position_embed = args['position_embed']
        self.gpt_posiiton_embed = args['gpt_position_embed']

        if args["pretrained_lm"] is not None:
            print("Loading from pretrained "+args["pretrained_lm"]+" model")
            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained(args["pretrained_lm"])
                if args["pretrained_lm"] != None:
                    if "image" in args["pretrained_lm"]:
                        #import decision_transformer
                        self.transformer_model = iGPT2LMHeadModel_LoRA.from_pretrained(
                            args["pretrained_lm"],
                            config=config,
                        )
                    elif "gpt" in args["pretrained_lm"]:
                        self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                            args["pretrained_lm"],
                            config=config,
                        )

                        p_dict = dict(self.transformer_model.named_parameters())

                        # if args["reinit_markov_head"]:
                        #     #for h_id in [1, 3, 4, 5]: # These heads has most attention assigned the current token
                        #     for h_id in [0, 2, 6, 7]:
                        #         #print(f'{p_dict["transformer.h.0.attn.c_attn.bias"]}') #[h_id*768:(h_id*768 + 10)]}')
                        #         init.zeros_(p_dict['transformer.h.0.attn.c_attn.bias'][h_id*64:(h_id+1)*64])
                        #         c_dict = dict(self.transformer_model.named_parameters())
                        #         #print(f'{c_dict["transformer.h.0.attn.c_proj.bias"][h_id*768:(h_id*768 + 10)]}')
                        #         init.xavier_normal_(p_dict['transformer.h.0.attn.c_attn.weight'][:, h_id*64:(h_id+1)*64])
                        #     print("Initialized part of the first attention successfully!")
                    else:
                        raise NotImplementedError

            elif "gpt" in args["pretrained_lm"]:
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                config.resid_pdrop = args["dropout"]
                self.transformer_model = GPT2LMHeadModel.from_pretrained(
                    args["pretrained_lm"],
                    config=config,
                )
            elif "llama" in args["pretrained_lm"]:
                #config = xx
                #self.transformer_model = LlamaForCausalLM.from_pretrained(
                self.transformer_model = LlamaModel.from_pretrained(
                    args["pretrained_lm"],
                    low_cpu_mem_usage=True,
                    #torch_dtype=torch.float16,
                    #load_in_4bit=True,
                )
                self.transformer_model.config.use_cache = False
                config = self.transformer_model.config

            if "gpt" in args["pretrained_lm"]: # gpt config
                hidden_size = config.n_embd
                self.hidden_size = config.n_embd
            else:                  # llama
                hidden_size = config.hidden_size
                self.hidden_size = config.hidden_size

        else:

            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained("gpt2")
                self.transformer_model = GPT2LMHeadModel_LoRA(config)
            else:
                config = transformers.GPT2Config(
                    n_embd=hidden_size,
                    **kwargs
                )
                # config = transformers.GPT2Config.from_pretrained("gpt2")
                # config.resid_pdrop = args["dropout"]
                # NOTE: If you comment two lines above, then we adopt non-pretrained 3-layer DT; otherwise we use the same config as the pretrained gpt2 model, but with random weights
                self.transformer_model = GPT2LMHeadModel(config)
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        # if max_ep_len > config.n_positions and args["extend_positions"]:
        #     current_max_pos, embed_size = self.transformer.wpe.weight.shape
        #     new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
        #         max_ep_len, embed_size
        #     )
        #     # copy position embeddings over and over to initialize the new position embeddings
        #     orig_k = 2
        #     k = orig_k
        #     step = current_max_pos - k
        #     new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
        #     while k < max_ep_len - 1:
        #         new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
        #             orig_k : min(max_ep_len - k + orig_k, current_max_pos)
        #         ]
        #         k += step
        #     self.transformer.wpe.weight.data = new_encoder_pos_embed

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        if args["mlp_embedding"]:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["mlp_embedding"]:
          if args["share_input_output_proj"]: raise ValueError("With MLP in embeddings, you cannot share the projections")
          self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
          self.predict_action = MLPBlock(self.hidden_size, self.act_dim, self.hidden_size)
          self.predict_return = torch.nn.Linear(hidden_size, 1)
        else:
          if args["share_input_output_proj"]:
            self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
          else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            self.predict_return = torch.nn.Linear(hidden_size, 1)

        if args["mgdt_sampling"]:
            self.predict_rtg = torch.nn.Linear(hidden_size, int(args["num_bins"]))
        else:
            self.predict_rtg = lambda x: None
        self.past_key_values = None
        print(self)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
        test=False,
    ):
        # print(f"{states.shape=}, {actions.shape=}, {rewards.shape=}, {returns_to_go.shape=}")
        ## states [64, 20, 11]
        ## actions [64, 20, 3]
        ## rewards.shape=torch.Size([64, 20, 1]),
        ## returns_to_go.shape=torch.Size([64, 20, 1])

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(f'{timesteps[0]=}') # consecutive numbers
        # print(f"{state_embeddings.shape=}, {time_embeddings.shape=}")
        ## both are ([64, 20, 768])

        # action_prototype_embeddings = self.action_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # abstract_action_embedding = self.action_abstraction_layer(action_embeddings, action_prototype_embeddings, action_prototype_embeddings)
        # returns_prototype_embeddings = self.returns_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # abstract_returns_embedding = self.returns_abstraction_layer(returns_embeddings, returns_prototype_embeddings, returns_prototype_embeddings)

        # print(f"{prototype_embeddings.shape=}, {abstract_state_embedding.shape=}")
        ## [1000, 768]), abstract_state_embedding.shape=torch.Size([64, 20, 768])

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        #abstract_embedding = self.state_abstraction_layer(stacked_inputs, state_prototype_embeddings, state_prototype_embeddings)

        #all_embs = self.embed_ln(stacked_inputs + abstract_embedding)
        all_embs = self.embed_ln(stacked_inputs)

        if self.position_embed:
            stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)
        else:
            stacked_inputs = all_embs

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
            to_add_position_embeds=self.gpt_posiiton_embed,
            output_attentions=True,
            output_hidden_states=True,
            interrupt_h=self.args["hidden_index"],
        )

        x = transformer_outputs["hidden_states"][self.args["hidden_index"]]

        #print(f"{type(transformer_outputs['attentions'][0])}") # tuple
        #print(f"{transformer_outputs['attentions'][0].shape}") # 12
        #print(f"{transformer_outputs['attentions'].keys()}")
        # if self.args["visualize_attn"] and transformer_outputs['attentions'][0].shape[-1] == 60:
        #     #plot attention
        #     # transformer_outputs['attentions'] has the shape of [n_layer, batch_size, n_head(12), seq_len, seq_len]
        #     for i in range(12):
        #         target_att = transformer_outputs['attentions'][0][0][i].cpu().detach()
        #         target_att = target_att[1::3]
        #         plt.imshow(target_att, cmap="hot")
        #         plt.savefig(f"test_{i}.png")

        #     # def show_attn_dist(attn):
        #     #     # to prove attention matrix distance does not matter!
        #     #     dist = torch.zeros((attn[0].shape[2], attn[0].shape[3]))
        #     #     for i in range(dist.shape[0]):
        #     #         for j in range(dist.shape[1]):
        #     #             dist[i][j] = abs(i - j)
            
        #     #     for i in range(len(attn)):
        #     #         layer_attn = attn[i].mean(0)[0].detach().cpu()
        #     #         attn_dist = (layer_attn * dist).mean()
        #     #         print(f"layer {i} attention distance: {attn_dist}")
        #     # show_attn_dist(transformer_outputs['attentions'])
        #     raise NotImplementedError

        #print(transformer_outputs.keys())
        self.past_key_values = transformer_outputs["past_key_values"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        observation_preds = None

        # after tanh, the predicted action is in range [-1, 1], to make use of OOD, we multiply it with `1.1`
        action_preds = 1.1 * self.predict_action(x[:, 1])  # predict next action given state
        #rgt_preds = self.predict_rtg(x[:, 0])
        return observation_preds, action_preds, None, transformer_outputs['attentions']

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
            #print(f"{timesteps=}")
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]

    def get_action_batch(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        n_envs=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model
        
        states = states.reshape(n_envs, -1, self.state_dim)
        actions = actions.reshape(n_envs, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(n_envs, -1, 1)
        timesteps = timesteps.reshape(n_envs, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
            #print(f"{timesteps=}")
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]
