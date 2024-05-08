import numpy as np
import math
import torch
import torch.nn as nn
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

class ContextDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    @property
    def transformer(self):
        return self.transformer_model.transformer

    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        prefix_text,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.do_reprograming = args["reprogram"]
        self.position_embed = args['position_embed']
        self.gpt_posiiton_embed = args['gpt_position_embed']
        self.trajectory_example = args['trajectory_example']
        
        if args["pretrained_lm"] is not None:
            print("Loading from pretrained "+args["pretrained_lm"]+" model")
            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained(args["pretrained_lm"])
                self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                    args["pretrained_lm"],
                    config=config
                )
            else:
                config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                config.resid_pdrop = args["dropout"]
                self.transformer_model = GPT2LMHeadModel.from_pretrained(
                    args["pretrained_lm"],
                    config=config,
                )
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

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

        if max_ep_len > config.n_positions and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed

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

        if self.do_reprograming:
            self.word_embeddings = self.transformer_model.get_input_embeddings().weight.clone().to(args["device"])

            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.state_prototype_mapping = nn.Linear(self.vocab_size, self.num_tokens)
            #self.action_prototype_mapping = nn.Linear(self.vocab_size, self.num_tokens)
            #self.returns_prototype_mapping = nn.Linear(self.vocab_size, self.num_tokens)

            self.state_abstraction_layer = StateAbstractionLayer(d_model=hidden_size, n_heads=8, d_keys=None, d_llm=hidden_size)
            #self.action_abstraction_layer = StateAbstractionLayer(d_model=hidden_size, n_heads=8, d_keys=None, d_llm=hidden_size)
            #self.returns_abstraction_layer = StateAbstractionLayer(d_model=hidden_size, n_heads=8, d_keys=None, d_llm=hidden_size)


        self.past_key_values = None
        print(self)

        self.word_embedding_layer = self.transformer_model.get_input_embeddings()#.clone()
        self.prefix_text = "<|start_task_description|>" + prefix_text + "<|end_task_description|>"
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.prefix_tokens = self.tokenizer.tokenize(self.prefix_text)
        self.prefix_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.prefix_tokens)).to(args["device"])
        # 106 tokens

        if self.trajectory_example:
            #self.example_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
            self.example_start_text = "<|start_examples|>"
            self.example_start_tokens = self.tokenizer.tokenize(self.example_start_text)
            self.example_start_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.example_start_tokens)).to(args["device"])

            self.example_end_text = "<|end_examples|>"
            self.example_end_tokens = self.tokenizer.tokenize(self.example_end_text)
            self.example_end_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(self.example_end_tokens)).to(args["device"])

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        e_states=None,
        e_actions=None,
        e_rewards=None,
        e_returns_to_go=None,
        e_timesteps=None,
        attention_mask=None,
        past_key_values=None,
        test=False,
    ):
        # print(f"{states.shape=}, {actions.shape=}, {rewards.shape=}, {returns_to_go.shape=}")
        ## states.shape=torch.Size([64, 20, 11]),
        ## actions.shape=torch.Size([64, 20, 3]),
        ## rewards.shape=torch.Size([64, 20, 1]),
        ## returns_to_go.shape=torch.Size([64, 20, 1])
        #if attention_mask != None:
        #    print(f"{attention_mask.shape=}")

        prefix_embed = self.prefix_text
        if self.trajectory_example:
            if states.shape[1] <= 5 or e_states == None:
                threshold_time_step = 0
            else:
                threshold_time_step = np.random.randint(20)
        else:
            threshold_time_step = 0
        if test:
            # consider pre-defined threshold_time_step
            pass

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
            #real_seq_len = seq_length + len(self.prefix_tokens)
            #print(f"{real_seq_len=}")

        prefix_mask = torch.ones((batch_size, len(self.prefix_tokens)), dtype=torch.long, device=states.device)

        # print(f"{attention_mask.shape=}") # [64, 20], with prefix: [64, 126]
        # print(f"{attention_mask[0, :]=}")

        prefix_embeddings = self.word_embedding_layer(self.prefix_ids) # [106, 768]
        batched_prefix_embeddings = torch.stack([prefix_embeddings for _ in range(batch_size)], dim=0)
        #print(f"{batched_prefix_embeddings.shape=}") # [106, 20, 768]

        if self.trajectory_example:
            example_start_embeddings = self.word_embedding_layer(self.example_start_ids)
            batched_example_start_embeddings = torch.stack([example_start_embeddings for _ in range(batch_size)], dim=0)
            example_end_embeddings = self.word_embedding_layer(self.example_end_ids)
            batched_example_end_embeddings = torch.stack([example_end_embeddings for _ in range(batch_size)], dim=0)
            if threshold_time_step > 0:
                e_state_embeddings = self.embed_state(e_states)
                e_action_embeddings = self.embed_action(e_actions)
                e_returns_embeddings = self.embed_return(e_returns_to_go)
                e_time_embeddings = self.embed_timestep(e_timesteps)
            

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        # print(f'{timesteps[0]=}') # consecutive numbers
        # print(f"{state_embeddings.shape=}, {time_embeddings.shape=}")
        ## both are ([64, 20, 768])

        # if self.do_reprograming:
        #     state_prototype_embeddings = self.state_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
        #     abstract_state_embeddings = self.state_abstraction_layer(state_embeddings, state_prototype_embeddings, state_prototype_embeddings)
        #     state_embeddings += abstract_state_embeddings

        #     action_prototype_embeddings = self.action_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
        #     abstract_action_embeddings = self.action_abstraction_layer(action_embeddings, action_prototype_embeddings, action_prototype_embeddings)
        #     action_embeddings += abstract_action_embeddings

        #     returns_prototype_embeddings = self.returns_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
        #     abstract_returns_embeddings = self.returns_abstraction_layer(returns_embeddings, returns_prototype_embeddings, returns_prototype_embeddings)
        #     returns_embeddings += abstract_returns_embeddings

        # print(f"{prototype_embeddings.shape=}, {abstract_state_embedding.shape=}")
        ## [1000, 768]), abstract_state_embedding.shape=torch.Size([64, 20, 768])

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = (
            torch.stack(
                (returns_embeddings,
                 state_embeddings,
                 action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        if self.do_reprograming:
            state_prototype_embeddings = self.state_prototype_mapping(self.word_embeddings.permute(1, 0)).permute(1, 0)
            abstract_stacked_inputs = self.state_abstraction_layer(stacked_inputs, state_prototype_embeddings, state_prototype_embeddings)
            stacked_inputs = abstract_stacked_inputs
        #abstract_embedding = self.state_abstraction_layer(stacked_inputs, state_prototype_embeddings, state_prototype_embeddings)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask,
                         attention_mask,
                         attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        all_embs = self.embed_ln(stacked_inputs)

        if self.position_embed:
            stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)
        else:
            stacked_inputs = all_embs

        if self.trajectory_example:
            if threshold_time_step > 0:
                example_embed_timestep = self.embed_timestep(timesteps[:, :threshold_time_step])
                stacked_examples = (
                    torch.stack(
                        (e_returns_embeddings[:, :threshold_time_step, :],
                        e_state_embeddings[:, :threshold_time_step, :],
                        e_action_embeddings[:, :threshold_time_step, :]), dim=1
                    )
                    .permute(0, 2, 1, 3)
                    .reshape(batch_size, 3 * threshold_time_step, self.hidden_size)
                )
                stacked_examples = self.embed_ln(stacked_examples)
                # batched_example_embeddings = torch.cat([
                #     batched_example_start_embeddings,
                #     e_returns_embeddings[:, :threshold_time_step, :] + example_embed_timestep,
                #     e_state_embeddings[:, :threshold_time_step, :] + example_embed_timestep, 
                #     e_action_embeddings[:, :threshold_time_step, :] + example_embed_timestep,
                #     batched_example_end_embeddings,], dim=1          
                # )
                batched_example_embeddings = torch.cat([
                    batched_example_start_embeddings,
                    stacked_examples,
                    batched_example_end_embeddings,], dim=1          
                )
            else:
                example_embed_timestep = self.embed_timestep(timesteps[:, :threshold_time_step])
                batched_example_embeddings = torch.cat([
                    batched_example_start_embeddings,
                    batched_example_end_embeddings,], dim=1          
                )
            example_mask = torch.ones((batch_size, batched_example_embeddings.shape[1]), dtype=torch.long, device=states.device)

            #print(f"{batched_prefix_embeddings.shape=}, {batched_example_embeddings.shape=}, {all_embs.shape=}")
            stacked_inputs = torch.cat([batched_prefix_embeddings, batched_example_embeddings, all_embs], dim=1)
            stacked_attention_mask = torch.cat([prefix_mask, example_mask, stacked_attention_mask], dim=1)
            #print(f"{stacked_attention_mask.shape=}")
            token_wise_threshold = len(self.prefix_ids) + len(self.example_start_ids) \
                + len(self.example_end_ids) + threshold_time_step * 3

        else: # no example module
            stacked_inputs = torch.cat([batched_prefix_embeddings, all_embs], dim=1)
            stacked_attention_mask = torch.cat([prefix_mask, stacked_attention_mask], dim=1)
            token_wise_threshold = len(self.prefix_ids)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
            to_add_position_embeds=self.gpt_posiiton_embed,
        )
        x = transformer_outputs["last_hidden_state"][:, token_wise_threshold:, :]
        self.past_key_values = transformer_outputs["past_key_values"]
        #print(f"{x.shape=}")# [64, 60, 768]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        observation_preds = None
        action_preds = self.predict_action(x[:, 1])  # predict next action given state
        return observation_preds, action_preds, None, None

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
