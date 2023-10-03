"""
Action Prompt Network
"""

import torch
from torch import nn

from models.HOIPrompting import MulitHeadAttention, HOIPrompt 
# from models.args import get_args

# args = get_args()
# device = torch.device('cuda', args.local_rank)


class PositionEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pos_embd = 0

        return pos_embd


class ActionPrompt(nn.Module):
    def __init__(self, num_actions = 50, dim = 1024):
        super().__init__()
        self.num_actions = num_actions
        self.dim = dim
        self.action_prompt_model = HOIPrompt(embed_dim = self.dim)
        self.atten = MulitHeadAttention(dim = self.dim, num_heads = 1)

    def forward(self, comb_fea: list, action_fea: list):
        print("comb_fea shape: ", comb_fea.shape)
        _, num_comb, _ = comb_fea.shape
        _, num_act, _ = action_fea.shape
        total_vis_prompt = []
        # total_action_prompt = []
        # total_action_prompt = torch.zeros(self.dim).to(device)
        total_action_prompt = torch.zeros((1, self.dim)).cuda()

        # generate visual prompts for each human-object combination
        for i in range(num_comb):
            single_comb_fea = comb_fea[:, i, :].unsqueeze(0)
            # print(single_comb_fea.shape)  # torch.Size([1, 19, 64])
            single_visual_prompt = self.action_prompt_model(torch.Tensor(action_fea), single_comb_fea)
            # print("single_visual_prompt shape: ", single_visual_prompt.shape) # torch.Size([1, 50, 64])
            total_vis_prompt.append(single_visual_prompt) # [torch.Size([1, 50, 64]), ...]

        # generate action prompts for each action
        for i in range(num_act):
            single_act_fea = action_fea[:, i, :].unsqueeze(0)
            # learned_act_prompt = torch.zeros((1, 1, self.dim)).to(device)
            # avg_act_prompt = torch.zeros(self.dim).to(device)
            avg_act_prompt = torch.zeros(self.dim).cuda()
            for j in range(num_comb):
                single_act_prompt = self.atten(
                    total_vis_prompt[j][:, i, :].unsqueeze(0),
                    single_act_fea, 
                    single_act_fea
                ).squeeze(0).squeeze(0)
                # print("single_act_prompt: ", single_act_prompt.shape) # torch.Size([1, 1, 64])
                avg_act_prompt += single_act_prompt
            avg_act_prompt /= num_comb
            # print("avg_act_prompt: ", avg_act_prompt)
            # # total_action_prompt.append(avg_act_prompt)
            # if i == 0:
            #     total_action_prompt = torch.stack((total_action_prompt, avg_act_prompt))
            # else:
            #     total_action_prompt = torch.cat((total_action_prompt, avg_act_prompt.unsqueeze(0)))
            total_action_prompt = torch.cat((total_action_prompt, avg_act_prompt.unsqueeze(0)), dim = 0)

        del total_vis_prompt
        return total_action_prompt[1:]


if __name__ == '__main__':
    f_ac = torch.randn(1, 50, 1024)  
    f_cb = torch.randn(1, 130, 1024) 
    model = ActionPrompt()
    act_ptompt = model(f_cb, f_ac)
    print("act_ptompt: ", act_ptompt)  
    print(act_ptompt.shape) # torch.Size([50, 1024])