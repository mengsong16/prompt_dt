# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers

from prompt_dt.trajectory_gpt2 import GPT2Model

class PromptDecisionTransformer(nn.Module):

    def __init__(
            self,
            state_dim,
            act_dim,
            goal_dim,
            hidden_size,
            no_prompt,
            prompt_method,
            max_length=None,
            max_ep_len=4096, # max length of an episode
            action_tanh=True, # use tanh instead of relu for output action activation
            parallelize_transformer=False, # parallelize transformer or not
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.no_prompt = no_prompt
        self.prompt_method = prompt_method
        self.goal_dim = goal_dim

        # transformer configuration
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # change transformer to parallelize mode for metaworld big model
        if parallelize_transformer:
            self.transformer.parallelize()

        # input encoders
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        # embed stacked input
        self.embed_ln = nn.LayerNorm(hidden_size)
       
        # note: we don't predict states or returns for the paper, but we keep the return and state decoder here
        # input of decoder has batch size B*L (L = prompt_seq_length + input_seq_length)
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        # prompt encoders
        if self.no_prompt == False:
            if self.prompt_method == "traj_prompt":
                self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
                self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
                self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
                self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)
            elif self.prompt_method == "goal_prompt":
                self.goal_prompt_embed = torch.nn.Linear(self.goal_dim, hidden_size)
            else:
                print("Error: unknown prompt method")
                exit()


    # input: a sequence of (s,a,r,t) of length max_length
    # output: a sequence of predicted (s,a,r) of length max_length
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)  # [B,L,state_dim] --> [B,L,hidden_size]
        action_embeddings = self.embed_action(actions) # [B,L,action_dim] --> [B,L,hidden_size]
        returns_embeddings = self.embed_return(returns_to_go) # [B,L,1] --> [B,L,hidden_size]
        time_embeddings = self.embed_timestep(timesteps) # [B,L,1] --> [B,L,hidden_size]

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # before permutation: [batch_size, 3, seq_length, hidden_size] (dim 1 is a new dim)
        # after permutation: [batch_size, seq_length, 3, hidden_size]
        # after reshape: sequence length becomes [batch_size, 3*seq_length, hidden_size]
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        # embed the concatenated input
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        if prompt is not None:
            # process prompt the same as dt
            if self.prompt_method == "traj_prompt":
                prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
                prompt_seq_length = prompt_states.shape[1]
                prompt_state_embeddings = self.prompt_embed_state(prompt_states)
                prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
                if prompt_returns_to_go.shape[1] % 10 == 1:
                    prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
                else:
                    prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
                prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

                prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
                prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
                prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

                # after reshape: [batch_size, 3*prompt_seq_length, hidden_size]
                # e.g. train: [720=45*16, 15=3*5, 128], [32=2*16, 15=3*5, 128], [32=2*16, 60=3*20, 128]
                prompt_stacked_inputs = torch.stack(
                    (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
                ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

                # to make the attention mask fit the stacked inputs, have to stack it as well
                # [32, 5] --> [32, 15], attention mask has type double
                prompt_stacked_attention_mask = torch.stack(
                    (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
                ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)
            elif self.prompt_method == "goal_prompt":
                goal_prompts, prompt_attention_mask = prompt #[32, 1], [32, 1]
                prompt_inputs = self.goal_prompt_embed(goal_prompts) # [32, 128]
                prompt_inputs = torch.unsqueeze(prompt_inputs, dim=1) # [32, 128] --> [32, 1, 128]
                prompt_stacked_inputs = torch.cat((prompt_inputs, prompt_inputs, prompt_inputs), dim=1) # [32, 1, 128] --> [32, 3, 128]
                prompt_stacked_attention_mask = torch.cat((prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1) # [32, 1] --> [32, 3]
                
            else:
                print("Error: undefined prompt method")
                exit()

        # concatenate input sequence and prompt sequence
        # assume sample one prompt for each trajectory in the batch (happen for both train and evaluation)
        stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1) # [32, 75=60+15, 128], [32, 63=60+3, 128]
        stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1) # [32, 75=60+15], [32, 63=60+3]

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        # Transformer: input shape = output last hidden shape
        x = transformer_outputs['last_hidden_state'] # [32, 75, 128], [32, 63, 128]

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            # Without prompt: actual_length = seq_length
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            # With prompt: actual_length = seq_length + prompt_length
            # traj_prompt: [32, 25, 3, 128] --> [32, 3, 25, 128]
            # goal_prompt: [32, 21, 3, 128] --> [32, 3, 21, 128]
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3) 
        

        # note here all the prompts are left-appended to x, so only return the last [:, -seq_length:, :] corresponding to batch data
        # (Rtg, s, a) sequence --> get predictions
        # x[:,2] = x[:,2,:,:] i.e. [32, 21, 128] (not equal to x[:,1:3,:,:])
        # x[:,1] = x[:,1,:,:] i.e. [32, 21, 128]
        # decoder input: [B, L, hidden_dimension] (L = prompt_seq_length + input_seq_length)
        # decoder output: [B, L, action_dimension] (e.g. [32, 21, 6])
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict (next) prompt+return sequence given prompt+action feature sequence (given state and action?)
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]    # predict (next) prompt+state sequence given prompt+action feature sequence (given state and action?)
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]  # predict (next) prompt+action sequence given prompt+state feature sequence

        return state_preds, action_preds, return_preds

    # input a sequence of (r,s,t) with arbitrary length
    # only return the last predicted action
    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        # only use left max_length subsequence, left padding if shorter than that
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # only attend to the valid part (non padding part)
            # 0 - not attend, 1 - attend
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            # left pad all tokens to the given sequence length max_length
            # left pad state with 0 if shorter than max_length
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            # left pad action with 0 if shorter than max_length 
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            # left pad rtg with 0 if shorter than max_length
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            # left pad timestep with 0 if shorter than max_length
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        # use the whole input sequence, regardless of its length
        else:
            attention_mask = None

        # Note: prompt within kwargs
        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
