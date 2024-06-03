import sys
import numpy as np
import torch
import imageio
import os
import cv2
from moviepy.editor import ImageSequenceClip

from decision_transformer.models.utils import decode_return, expert_sampling, mgdt_logits

def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    record_video=False,
    video_path=None
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    frames = []

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1
        
        if record_video and t % 2 == 0:
            curr_frame = env.render(mode='rgb_array')
            print(curr_frame.shape)
            frame_resized = cv2.resize(curr_frame, (84, 84)) 
            frames.append(frame_resized)

        if done:
            break

    model.past_key_values = None

    if record_video:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path, logger=None)

    return episode_return, episode_length

def parallel_evaluate_episode_rtg(
    args,
    multi_envs,
    state_dim,
    act_dim,
    model,
    max_timesteps=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    context_len=20,
    record_video=False,
    video_path=None
):
    torch.set_printoptions(precision=10)
    with open("actions_reference/states.npy", "rb") as f:
        state_list_tmp = np.load(f)
    with open("actions_reference/rewards.npy", "rb") as f:
        reward_list_tmp = np.load(f)
    with open("actions_reference/dones.npy", "rb") as f:
        done_list_tmp = np.load(f)
    # "context_len" is the same as "max_length" in dt.py
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    #state = multi_envs.reset(seed=0)
    state = multi_envs.reset(seed=22)
    if args['action_analyze_no_interaction']:
        state = state_list_tmp[0]

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    num_envs = multi_envs.num_envs
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, 1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    #print(f"initial states: {states}")
    #print(f"given target return: {target_return}")
    ORIGINAL_RTG = target_return
    actions = torch.zeros((num_envs, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((num_envs, 0), device=device, dtype=torch.float32)
    target_returns = torch.tensor([target_return] * num_envs, device=device, dtype=torch.float32).reshape(num_envs, 1, 1)
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, 1)

    t, done_flags = 0, np.zeros(num_envs, dtype=bool)
    episode_returns, episode_lens = np.zeros(num_envs), np.zeros(num_envs)
    heatmap_list = []
    RTG_list = [target_return]
    # state_list_tmp = [state]
    reward_list_tmp = []
    # done_list_tmp = []
    ANALYSIZE_RTG_RANGE = 3600 # 361
    #ANALYSIZE_RTG_RANGE = 1
    kk_list = []
    while not done_flags.all() and t < max_timesteps:
        if args['action_analyze']:
            print(t)
            #print(f"{actions=}")
            tmp_actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
            #print(f"{tmp_actions=}")
            tmp_rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

            # batchify with batch size equals to the number of RTGs to evaluate
            tmp_states = torch.stack([states[0, -context_len:, :].to(dtype=torch.float32) for _ in range(ANALYSIZE_RTG_RANGE)], dim=0)
            tmp_rewards = torch.stack([tmp_rewards[0, -context_len:,].to(dtype=torch.float32) for _ in range(ANALYSIZE_RTG_RANGE)], dim=0)
            tmp_actions = torch.stack([tmp_actions[0, -context_len:, :].to(dtype=torch.float32) for _ in range(ANALYSIZE_RTG_RANGE)], dim=0)
            tmp_target_returns = torch.stack([target_returns[0, -context_len:-1, :].to(dtype=torch.float32) for _ in range(ANALYSIZE_RTG_RANGE)], dim=0)
            tmp_timesteps = torch.stack([timesteps[0, -context_len:] for _ in range(ANALYSIZE_RTG_RANGE)], dim=0)

            #tmp_target_returns = torch.stack([target_returns[0, -context_len:, :].to(dtype=torch.float32) for _ in range(361)], dim=0)
            # print(tmp_target_returns.shape) # [361, seq_len, 1]
            if t == 0:
                manual_RTGs = torch.arange(1, ANALYSIZE_RTG_RANGE + 1).to(dtype=torch.float32, device="cuda").unsqueeze(-1).unsqueeze(-1) / 1000.0
                #manual_RTGs = torch.ones((1, 1, 1)).to(dtype=torch.float32, device="cuda") * 1.8
                tmp_target_returns = manual_RTGs
            else:
                manual_RTGs = torch.arange(1, ANALYSIZE_RTG_RANGE + 1).to(dtype=torch.float32, device="cuda").unsqueeze(-1).unsqueeze(-1) / 1000.0
                #manual_RTGs = torch.ones((1, 1, 1)).to(dtype=torch.float32, device="cuda") * pred_returns
                tmp_target_returns = torch.cat([tmp_target_returns, manual_RTGs], dim=1)

            if args['quantize_rtg']:
                _, predict_actions_rtg, _, _ = model.forward(
                    ((tmp_states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                    tmp_actions[:, -context_len:, :].to(dtype=torch.float32),
                    tmp_rewards.to(dtype=torch.float32),
                    (tmp_target_returns[:, -context_len:, :]*1000.0).to(int).to(dtype=torch.float32)/1000.0,
                    tmp_timesteps[:, -context_len:].to(dtype=torch.long),
                )
            else:
                _, predict_actions_rtg, _, _ = model.forward(
                ((tmp_states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                tmp_actions[:, -context_len:, :].to(dtype=torch.float32),
                tmp_rewards.to(dtype=torch.float32),
                tmp_target_returns[:, -context_len:, :].to(dtype=torch.float32),
                tmp_timesteps[:, -context_len:].to(dtype=torch.long),
                )
            # if t == 0 or t==1:
            #     print(f"\nINPUTS for search")
            #     print(f"{tmp_states[0]=}")
            #     print(f"{tmp_rewards[0]=}")
            #     print(f"{tmp_target_returns[:]}") # 1799
            #     #print(tmp_timesteps[0])
            #     #print(f"{predict_actions_rtg[:][-1]=}") # 1799
            #     print(f"{predict_actions_rtg=}")
            #     print(f"{manual_RTGs[:]=}") # 1799
            #     if t == 1:
            #         raise NotImplementedError
            #print(f"produced action: {tmp_action.shape=}")
            tmp_last_action = predict_actions_rtg[:, -1]
            a = None #tmp_last_action[361]
            a_find = False

            action_array = predict_actions_rtg[:, -1].to("cpu").numpy()
            if args['search_rtg']:
                if t == 0:
                    kk = int(target_return*1000)-1
                    a_find = True
                else:
                    for i in range(3600-11, 0, -1):
                        #if tmp_last_action[kk][0] <= 1.0 and tmp_last_action[kk][0] > tmp_last_action[kk + 1][0]:
                        d = tmp_last_action[i][0] - tmp_last_action[i + 10][0]
                        if d > 0.1:
                            #print(f"{tmp_last_action[kk][0]=}, {tmp_last_action[kk + 1][0]=}")
                            kk = i+10
                            a_find = True
                            break
                        # if tmp_last_action[kk][0] > tmp_last_action[kk + 1][0]:
                        #     #print(f"{tmp_last_action[kk][0]=}, {tmp_last_action[kk + 1][0]=}")
                        #     a = tmp_last_action[kk]
                        #     print(f"{kk=}")
                        #     break

                a = tmp_last_action[kk]
                if not a_find:
                    print(f"kk not found!")
                    #print(f"{tmp_last_action[:, 0]}")
                    kk = -1
                kk_list.append(kk)

            with open(f"actions/actions_{t}.npy", 'wb') as f:
                np.save(f, action_array)

        # print(f"{target_returns[0][-1]=}")
        # print(f"{states[0][-1]=}")


        #print(f"{actions=}")
        actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
        #print(f"{actions=}")

        rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

        # _, action, rtg_prediction, attn_hm = model.forward(
        #     ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
        #     actions[:, -context_len:, :].to(dtype=torch.float32),
        #     rewards.to(dtype=torch.float32),
        #     (target_returns[:, -context_len:, :]*1000.0).to(int).to(dtype=torch.float32)/1000.0,
        #     timesteps[:, -context_len:].to(dtype=torch.long),
        # )
        # action = action[:, -1]

        if not args['search_rtg']:
            _, action, rtg_prediction, attn_hm = model.forward(
                ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                actions[:, -context_len:, :].to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                (target_returns[:, -context_len:, :]*1000.0).to(int).to(dtype=torch.float32)/1000.0,
                timesteps[:, -context_len:].to(dtype=torch.long),
            )
            action = action[:, -1]
            # if t == 0 or t==1:
            #     print(t)
            #     print("\n Input")
            #     print(states[:, -context_len:, :].to(dtype=torch.float32))
            #     print(actions[:, -context_len:, :].to(dtype=torch.float32))
            #     print(rewards.to(dtype=torch.float32))
            #     print((target_returns[:, -context_len:, :]*1000.0).to(int).to(dtype=torch.float32)/1000.0)
            #     print(timesteps[:, -context_len:].to(dtype=torch.long))
            #     print(action)
            #     if t == 1:
            #         raise NotImplementedError
        if args["visualize_attn"]:
            heatmap_list.append(attn_hm)

        if args["mgdt_sampling"]:
            opt_rtg = decode_return(
                args["env"],
                expert_sampling(
                    mgdt_logits(rtg_prediction),
                    top_percentile=args["top_percentile"],
                    expert_weight=args["expert_weight"],
                ),
                num_bin=args["num_bins"],
                rtg_scale=args["rtg_scale"],
            )

            _, action, _, _ = model.forward(
                ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                actions[:, -context_len:, :].to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                opt_rtg,
                timesteps[:, -context_len:].to(dtype=torch.long),
            )

        #predicted_action = action
        
        #print(f"{kk=}, d_action: {a[0].detach().cpu().numpy() - action[0].detach().cpu().numpy()}")

        #d_a = (tmp_last_action[int(target_returns[0][-1][0].cpu().item()*1000) - 1] - action).cpu().numpy()

        # if t > 300:
        #     action = a
        #action = a
        # if t > 100:
        #     action = tmp_action[int(target_returns[0][-1][0].cpu().item()*1000)][-1]
        if args['action_analyze'] and args['search_rtg']:
            #action = predict_actions_rtg[int(target_returns[0][-1][0].cpu().item()*1000) - 1][-1].unsqueeze(0)
            if kk == -1: # if the searching does not find an action, use the RTG
                action = predict_actions_rtg[int(target_returns[0][-1][0].cpu().item()*1000) - 1][-1].unsqueeze(0)
            else:
                action = a.unsqueeze(0)
            # to test manually set the batch size to 1 for RTG search
            #action = predict_actions_rtg[0][-1].unsqueeze(0)

        #print(f"{manual_RTGs[int(target_returns[0][-1][0].cpu().item()*1000) - 1]} {RTG_list[-1]} {d_a}")

        actions[:, -1, :] = action
        action = action.detach().cpu().numpy()
        #print(f"Tensor action {action}")
        
        #print(f"np action {action}")

        if args['action_analyze_no_interaction']:
            # analyze the action selection with stored states
            state, reward, done = state_list_tmp[t+1], reward_list_tmp[t], done_list_tmp[t]
        else:
            if args['eval_only']:
                print(f"actions for env: {action}")
            state, reward, done, _ = multi_envs.step(action)
        # state_list_tmp.append(state)
        reward_list_tmp.append(float(reward[0]))
        # done_list_tmp.append(done)

        cur_state = torch.from_numpy(state).to(device=device).reshape(num_envs, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        rewards[:, -1] = torch.tensor(reward, dtype=torch.float32)
        episode_returns += reward * ~done_flags
        episode_lens += 1 * ~done_flags
        done_flags = np.bitwise_or(done_flags, done)

        if mode != "delayed":
            pred_returns = target_returns[:, -1, 0] - torch.tensor((reward / scale), device=device, dtype=torch.float32)
        else:
            pred_returns = target_returns[:, -1, 0]
        if args['action_analyze']:
            RTG_list.append(pred_returns.detach().cpu().item())
        target_returns = torch.cat([target_returns, pred_returns.reshape(num_envs, 1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )
        t += 1
    if args['eval_only']:
        print(repr(reward_list_tmp))
    if args['action_analyze']:
        with open("actions/rtg.npy", 'wb') as f:
           np.save(f, np.array(RTG_list))

        with open("actions/kk.npy", 'wb') as f:
           np.save(f, np.array(kk_list))
        # with open("actions/states.npy", 'wb') as f:
        #    np.save(f, np.array(state_list_tmp))
        # with open("actions/rewards.npy", 'wb') as f:
        #    np.save(f, np.array(reward_list_tmp))
        # with open("actions/dones.npy", 'wb') as f:
        #    np.save(f, np.array(done_list_tmp))
        print(f"{episode_returns=}")
        raise NotImplementedError
    return episode_returns, episode_lens

def parallel_evaluate_episode_rtg_(
    args,
    multi_envs,
    state_dim,
    act_dim,
    model,
    max_timesteps=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    context_len=20,
    record_video=False,
    video_path=None
):
    # "context_len" is the same as "max_length" in dt.py
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    #state = multi_envs.reset(seed=0)
    #state = multi_envs.reset()

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    num_envs = multi_envs.num_envs
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, 1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((num_envs, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((num_envs, 0), device=device, dtype=torch.float32)
    target_returns = torch.tensor([target_return] * num_envs, device=device, dtype=torch.float32).reshape(num_envs, 1, 1)
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, 1)

    t, done_flags = 0, np.zeros(num_envs, dtype=bool)
    episode_returns, episode_lens = np.zeros(num_envs), np.zeros(num_envs)
    heatmap_list = []
    RTG_list = [target_return]
    # state_list_tmp = [state]
    # reward_list_tmp = []
    # done_list_tmp = []
    while not done_flags.all() and t < max_timesteps:
        if args['action_analyze']:
            print(t)
            tmp_actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
            tmp_rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

            #print(f"{states.shape=}")
            # batchify
            tmp_states = torch.stack([states[0, -context_len:, :].to(dtype=torch.float32) for _ in range(361)], dim=0)
            tmp_rewards = torch.stack([tmp_rewards[0, -context_len:,].to(dtype=torch.float32) for _ in range(361)], dim=0)
            tmp_actions = torch.stack([tmp_actions[0, -context_len:, :].to(dtype=torch.float32) for _ in range(361)], dim=0)
            tmp_target_returns = torch.stack([target_returns[0, -context_len:-1, :].to(dtype=torch.float32) for _ in range(361)], dim=0)
            tmp_timesteps = torch.stack([timesteps[0, -context_len:].to(dtype=torch.float32) for _ in range(361)], dim=0)
            #tmp_target_returns = torch.stack([target_returns[0, -context_len:, :].to(dtype=torch.float32) for _ in range(361)], dim=0)
            # print(tmp_target_returns.shape) # [361, seq_len, 1]
            if t == 0:
                tmp_target_returns = torch.arange(1, 362).to(dtype=torch.float32, device="cuda").unsqueeze(-1).unsqueeze(-1) / 100.0
            else:
                manual_RTGs = torch.arange(1, 362).to(dtype=torch.float32, device="cuda").unsqueeze(-1).unsqueeze(-1) / 100.0
                tmp_target_returns = torch.cat([tmp_target_returns, manual_RTGs], dim=1)


            _, tmp_action, _, _ = model.forward(
                ((tmp_states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                tmp_actions[:, -context_len:, :].to(dtype=torch.float32),
                tmp_rewards.to(dtype=torch.float32),
                tmp_target_returns[:, -context_len:, :].to(dtype=torch.float32),
                tmp_timesteps[:, -context_len:].to(dtype=torch.long),
            )
            #print(f"produced action: {action.shape=}")
            action_array = tmp_action[:, -1].to("cpu").numpy()
            with open(f"actions/actions_{t}.npy", 'wb') as f:
                np.save(f, action_array)

        # print(f"{target_returns[0][-1]=}")
        # print(f"{states[0][-1]=}")
        actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

        _, action, rtg_prediction, attn_hm = model.forward(
            ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
            actions[:, -context_len:, :].to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_returns[:, -context_len:, :].to(dtype=torch.float32),
            timesteps[:, -context_len:].to(dtype=torch.long),
        )
        if args["visualize_attn"]:
            heatmap_list.append(attn_hm)

        if args["mgdt_sampling"]:
            opt_rtg = decode_return(
                args["env"],
                expert_sampling(
                    mgdt_logits(rtg_prediction),
                    top_percentile=args["top_percentile"],
                    expert_weight=args["expert_weight"],
                ),
                num_bin=args["num_bins"],
                rtg_scale=args["rtg_scale"],
            )

            _, action, _, _ = model.forward(
                ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
                actions[:, -context_len:, :].to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                opt_rtg,
                timesteps[:, -context_len:].to(dtype=torch.long),
            )

        action = action[:, -1]
        actions[:, -1, :] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = multi_envs.step(action)
        state_list_tmp.append(state)
        reward_list_tmp.append(reward)
        done_list_tmp.append(done)

        cur_state = torch.from_numpy(state).to(device=device).reshape(num_envs, 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        rewards[:, -1] = torch.tensor(reward, dtype=torch.float32)
        episode_returns += reward * ~done_flags
        episode_lens += 1 * ~done_flags
        done_flags = np.bitwise_or(done_flags, done)

        if mode != "delayed":
            pred_returns = target_returns[:, -1, 0] - torch.tensor((reward / scale), device=device, dtype=torch.float32)
        else:
            pred_returns = target_returns[:, -1, 0]
        if args['action_analyze']:
            RTG_list.append(pred_returns.to("cpu").item())
        target_returns = torch.cat([target_returns, pred_returns.reshape(num_envs, 1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )
        t += 1
    if args['action_analyze']:
        with open("actions/rtg.npy", 'wb') as f:
           np.save(f, np.array(RTG_list))
        
        with open("actions/states.npy", 'wb') as f:
           np.save(f, np.array(state_list_tmp))
        with open("actions/rewards.npy", 'wb') as f:
           np.save(f, np.array(reward_list_tmp))
        with open("actions/dones.npy", 'wb') as f:
           np.save(f, np.array(done_list_tmp))
        print(f"{episode_returns=}")
        raise NotImplementedError
    return episode_returns, episode_lens


# def parallel_evaluate_episode_rtg(
#     args,
#     multi_envs,
#     state_dim,
#     act_dim,
#     model,
#     max_timesteps=1000,
#     scale=1000.0,
#     state_mean=0.0,
#     state_std=1.0,
#     device="cuda",
#     target_return=None,
#     mode="normal",
#     context_len=20,
#     record_video=False,
#     video_path=None
# ):
#     # "context_len" is the same as "max_length" in dt.py
#     model.eval()
#     model.to(device=device)

#     state_mean = torch.from_numpy(state_mean).to(device=device)
#     state_std = torch.from_numpy(state_std).to(device=device)

#     #state = multi_envs.reset(seed=0)
#     state = multi_envs.reset()

#     if mode == "noise":
#         state = state + np.random.normal(0, 0.1, size=state.shape)

#     num_envs = multi_envs.num_envs
#     states = (
#         torch.from_numpy(state)
#         .reshape(num_envs, 1, state_dim)
#         .to(device=device, dtype=torch.float32)
#     )
#     actions = torch.zeros((num_envs, 0, act_dim), device=device, dtype=torch.float32)
#     rewards = torch.zeros((num_envs, 0), device=device, dtype=torch.float32)
#     target_returns = torch.tensor([target_return] * num_envs, device=device, dtype=torch.float32).reshape(num_envs, 1, 1)
#     timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, 1)

#     t, done_flags = 0, np.zeros(num_envs, dtype=bool)
#     episode_returns, episode_lens = np.zeros(num_envs), np.zeros(num_envs)
#     heatmap_list = []
#     while not done_flags.all() and t < max_timesteps:
#         actions = torch.cat([actions, torch.zeros((num_envs, 1, act_dim), device=device)], dim=1)
#         rewards = torch.cat([rewards, torch.zeros((num_envs, 1), device=device)], dim=1)

#         _, action, rtg_prediction, attn_hm = model.forward(
#             ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
#             actions[:, -context_len:, :].to(dtype=torch.float32),
#             rewards.to(dtype=torch.float32),
#             target_returns[:, -context_len:, :].to(dtype=torch.float32),
#             timesteps[:, -context_len:].to(dtype=torch.long),
#         )
#         if args["visualize_attn"]:
#             heatmap_list.append(attn_hm)

#         if args["mgdt_sampling"]:
#             opt_rtg = decode_return(
#                 args["env"],
#                 expert_sampling(
#                     mgdt_logits(rtg_prediction),
#                     top_percentile=args["top_percentile"],
#                     expert_weight=args["expert_weight"],
#                 ),
#                 num_bin=args["num_bins"],
#                 rtg_scale=args["rtg_scale"],
#             )

#             _, action, _, _ = model.forward(
#                 ((states[:, -context_len:, :].to(dtype=torch.float32) - state_mean) / state_std),
#                 actions[:, -context_len:, :].to(dtype=torch.float32),
#                 rewards.to(dtype=torch.float32),
#                 opt_rtg,
#                 timesteps[:, -context_len:].to(dtype=torch.long),
#             )

#         action = action[:, -1]
#         actions[:, -1, :] = action
#         action = action.detach().cpu().numpy()

#         state, reward, done, _ = multi_envs.step(action)

#         cur_state = torch.from_numpy(state).to(device=device).reshape(num_envs, 1, state_dim)
#         states = torch.cat([states, cur_state], dim=1)
#         rewards[:, -1] = torch.tensor(reward, dtype=torch.float32)
#         episode_returns += reward * ~done_flags
#         episode_lens += 1 * ~done_flags
#         done_flags = np.bitwise_or(done_flags, done)

#         if mode != "delayed":
#             pred_returns = target_returns[:, -1, 0] - torch.tensor((reward / scale), device=device, dtype=torch.float32)
#         else:
#             pred_returns = target_returns[:, -1, 0]
#         target_returns = torch.cat([target_returns, pred_returns.reshape(num_envs, 1, 1)], dim=1)
#         timesteps = torch.cat(
#             [timesteps, torch.ones((num_envs, 1), device=device, dtype=torch.long) * (t + 1)],
#             dim=1,
#         )
#         t += 1

#     # if n_envs == 1, iterate heat_map, get the last line for the attention
#     # calculate intra-time-step interaction, inter-time-step interaction for all time step, for only full contex len data
#     if args["visualize_attn"]:
#         # heatmap_list is a List[Tuple[Tensor]], len(List) is episdoe len. Len(Tuple) is n_layers;
#         # each heatmap is a Tensor [batch, n_head, seq_len, seq_len]

#         all_layer_list = []
#         n_head = heatmap_list[0][0].shape[1]
#         n_layer = len(heatmap_list[0])

#         print(f"{n_layer=}") # 12 layers
#         print(f"{heatmap_list[0][0].shape=}")

#         for layer_id in range(min(4, n_layer)): # n_layer

#             ret_list = []
#             for i in range(n_head): # iterate through heads

#                 last_row_list = []
#                 for hm_all in heatmap_list:
#                     #print(f"{hm_all[0].shape=}") # bs, n_head, seq, seq
#                     hm = hm_all[layer_id][0][i] # get the first layer and the first head

#                     if hm.shape[-1] == 60:
#                         last_row = hm[-2, :]

#                         #cum_last_row = np.cumsum(last_row.detach().cpu().numpy()[::-1]) # get rid of the last action, which is a zero tensor
#                         #step_wise_hm = cum_last_row[2::3] # step_wise_hm should be [20]
#                         last_row = last_row.detach().cpu().numpy()[::-1]
#                         step_wise_hm = last_row[1:]
#                         last_row_list.append(step_wise_hm)

#                 if (len(last_row_list) > 0):
#                     final_last_row = sum(last_row_list) / len(last_row_list)
#                     ret_list.append(final_last_row)
#                     #print(f"{final_last_row=}")
#                 else:
#                     print(f"{len(heatmap_list)=}")

#             all_layer_list.append(ret_list)
#         np.set_printoptions(threshold=sys.maxsize)
#         print(repr(np.array(all_layer_list)))
#     return episode_returns, episode_lens
