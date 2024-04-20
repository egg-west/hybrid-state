import numpy as np
import torch
import tqdm
import time
from itertools import cycle


class Trainer:
    def __init__(
        self,
        args,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        train_nlp_dataset=None,
        eval_nlp_dataset=None,
        scheduler=None,
        eval_fns=None,
        eval_only=False,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.step = 0
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.eval_only = eval_only
        self.eval_nlp_dataset = cycle(iter(eval_nlp_dataset))
        self.train_nlp_dataset = cycle(iter(train_nlp_dataset))

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        # lm_losses = []
        logs = dict()

        train_start = time.time()

        if not self.eval_only:
            self.model.train()
            mean_loss = 0
            progress_bar = tqdm.tqdm(range(num_steps), desc=f"Training")
            for _ in progress_bar:
                # train_loss, lm_loss = self.train_step()
                train_loss = self.train_step()
                train_losses.append(train_loss)
                # lm_losses.append(lm_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

                logs["time/training"] = time.time() - train_start
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)
                # logs["training/lm_loss_mean"] = np.mean(lm_losses)
                # logs["training/lm_loss_std"] = np.std(lm_losses)
                
                progress_bar.set_postfix({"loss": logs["training/train_loss_mean"], "lr": self.optimizer.param_groups[0]['lr']})
                

        eval_start = time.time()

        self.model.eval()
        for eval_fn in tqdm.tqdm(self.eval_fns, desc="Evaluating"):
            outputs = eval_fn(self.model)
            print(outputs)
            for k, v in outputs.items():
                print(k,":",v)
                logs[f"evaluation/{k}"] = v

        if not self.eval_only:
            logs["time/total"] = time.time() - self.start_time
        logs["time/evaluation"] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        if not self.eval_only:
            if self.args.get("outdir") and iter_num % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.args['outdir']}/model_{iter_num}.pt",
                )

        return logs

    def train_step(self):
        self.optimizer.zero_grad()
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        if self.args["fp16"]:
            with torch.cuda.amp.autocast():

                state_preds, action_preds, reward_preds = self.model.forward(
                    states,
                    actions,
                    rewards,
                    masks=None,
                    attention_mask=attention_mask,
                    target_return=returns,
                )

                # note: currently indexing & masking is not fully correct
                loss = self.loss_fn(
                    state_preds,
                    action_preds,
                    reward_preds,
                    state_target[:, 1:],
                    action_target,
                    reward_target[:, 1:],
                )
        elif self.args["inverse"]:
            state_preds, action_preds, reward_preds = self.model.forward(
                states,
                actions,
                rewards,
                masks=None,
                attention_mask=attention_mask,
                target_return=returns,
            )
            loss = torch.mean((action_preds - actions) ** 2) + torch.mean((state_preds[:, :-1] - states[:, 1:]) ** 2)
        else:

            state_preds, action_preds, reward_preds = self.model.forward(
                states,
                actions,
                rewards,
                masks=None,
                attention_mask=attention_mask,
                target_return=returns,
            )

            # note: currently indexing & masking is not fully correct
            loss = self.loss_fn(
                state_preds,
                action_preds,
                reward_preds,
                state_target[:, 1:],
                action_target,
                reward_target[:, 1:],
            )

        if self.args["fp16"]:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.detach().cpu().item()
