from asyncio import get_event_loop
import copy
import pickle as pickle
from random import sample
from sys import getallocatedblocks

from yaml import Token
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import datetime
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer
import clip
from transformers import (
    pipeline,
    ChineseCLIPProcessor,
    ChineseCLIPModel,
    AutoTokenizer,
    CLIPModel,
)
from googletrans import Translator

# from logger import Logger
import models_mae
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from timm.models.vision_transformer import Block
from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    PositionalEncodingPermute3D,
)


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, p, q):
        eps = 1e-10
        p, q = p + eps, q + eps
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Applying a small epsilon to avoid log(0)
        m = 0.5 * (p + q)
        log_m = m.log()
        kl_pm = self.kl(log_m, p.log())
        kl_qm = self.kl(log_m, q.log())
        jsd = 0.5 * (kl_pm + kl_qm).sum(dim=-1)
        return jsd


class InteractionModule(nn.Module):
    def __init__(self, unified_dim, agr_threshold, sem_threshold):
        agr_threshold = 0.3
        sem_threshold = 0.3
        balance_loss_coef = 0.1
        router_z_loss_coef = 0.01
        interaction_loss_coef = 0.7
        super(InteractionModule, self).__init__()
        self.jsd_module = JSD()
        self.kl = nn.KLDivLoss(reduction="none", log_target=True)
        # 256
        self.unified_dim = 64
        self.modality_attn = TokenAttention(self.unified_dim)
        self.soft_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 2),
        )

        self.agr_threshold = torch.tensor(agr_threshold, requires_grad=False)
        self.sem_threshold = torch.tensor(sem_threshold, requires_grad=False)
        # self.noisy_gate = False
        # self.noise_scale = 1.5
        # self.log_scale = None
        self.interaction_loss = nn.CrossEntropyLoss()
        self.interaction_loss_coef = interaction_loss_coef
        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def clip_similarity(self, text_embeds, image_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        clip_scores = (image_embeds * text_embeds).sum(dim=-1)
        return clip_scores

    def compute_router_z_loss(self, gate1_logits, gate2_logits):
        logits = torch.cat([gate1_logits], dim=1)
        max_logits = torch.logsumexp(logits, dim=1)
        router_z_loss = torch.mean(max_logits**2)
        return self.router_z_loss_coef * router_z_loss

    def compute_balance_loss(self, assignments):
        num_experts = 2
        batch_size = assignments.size(0)
        expert_counts = torch.zeros(num_experts)
        for i in range(num_experts):
            expert_counts[i] = (assignments == i).sum()
        distribution = expert_counts / batch_size
        target_distribution = torch.full_like(distribution, fill_value=1 / num_experts)
        balancing_loss = F.mse_loss(distribution, target_distribution)
        return balancing_loss

    def forward(self, p_t, p_i, e_t, e_i, m_t, m_i):
        # Compute supervision signal
        js_div = self.jsd_module(p_t, p_i)
        clip_score = self.clip_similarity(m_t, m_i)

        agr_gate_scores = (js_div < self.agr_threshold).type(torch.int64)
        sem_gate_scores = (clip_score > self.sem_threshold).type(torch.int64)

        stacked_features = torch.stack(
            (e_t, e_i, m_t, m_i),
            dim=1,
        )

        # unimodal_outputs = torch.cat((p_t, p_i), dim=1)
        # unimodal_features = self.agr_feature(unimodal_outputs)
        # multimodal_features = torch.cat((m_t, m_i), dim=1)
        # gate_inputs = torch.cat((unimodal_features, multimodal_features), dim=1)

        gate_inputs, _ = self.modality_attn(stacked_features)
        gate_logits = self.soft_gate(gate_inputs)
        targets = sem_gate_scores

        # agr_logits = gate_logits[:, :2]
        # sem_logits = gate_logits[:, 2:]
        interaction_loss = self.interaction_loss_coef * (
            self.interaction_loss(gate_logits, targets)
        )

        single_gate = torch.argmax(F.softmax(gate_logits, dim=1), dim=1)
        dispatch_index = single_gate
        router_z_loss = self.router_z_loss_coef * self.compute_router_z_loss(
            gate_logits, None
        )
        balance_loss = self.balance_loss_coef * self.compute_balance_loss(
            dispatch_index
        )
        expert_mask = F.softmax(gate_logits, dim=1)
        expert_mask = torch.cat(
            (expert_mask[:, 0:1].repeat(1, 2), expert_mask[:, 1:2].repeat(1, 2)), dim=1
        )

        gate_loss = interaction_loss + router_z_loss + balance_loss
        return expert_mask, gate_loss


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt(
            (
                torch.sum((x.permute([1, 0]) - self.mu(x)).permute([1, 0]) ** 2, (1))
                + 0.000000023
            )
            / (x.shape[1])
        )

    def forward(self, x, mu, sigma):
        """Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        # print(mu.shape) # 12
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean / x_std
        # print(x_mean.shape) # 768, 12
        return (sigma.squeeze(1) * (x_norm + mu.squeeze(1))).permute([1, 0])


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super(SimpleGate, self).__init__()
        self.dim = dim

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2


class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class VimoeAblation(nn.Module):
    def __init__(
        self,
        batch_size=64,
        dataset="weibo",
        text_token_len=197,
        image_token_len=197,
        is_use_bce=True,
        thresh=0.5,
        agr_threshold=0.3,
        sem_threshold=0.3,
        warmup_epochs=0,
    ):
        self.projection_only = False
        self.thresh = thresh
        self.batch_size = batch_size
        self.text_token_len, self.image_token_len = text_token_len, image_token_len
        model_size = "base"
        self.model_size = model_size
        self.dataset = dataset
        self.LOW_BATCH_SIZE_AND_LR = ["Twitter", "politi"]
        print("we are using adaIN")

        self.unified_dim, self.text_dim = 768, 768
        self.is_use_bce = is_use_bce
        out_dim = 1 if self.is_use_bce else 2
        self.num_expert = 2  # 2
        self.depth = 1  # 2
        super(VimoeAblation, self).__init__()

        # ================ Initialize models ================s
        # IMAGE SEMANTIC: MAE
        self.image_model = models_mae.__dict__[
            "mae_vit_{}_patch16".format(self.model_size)
        ](norm_pix_loss=False)
        checkpoint = torch.load(
            "./mae_pretrain_vit_{}.pth".format(self.model_size), map_location="cpu"
        )
        self.image_model.load_state_dict(checkpoint["model"], strict=False)

        # TEXT: BERT OR PRETRAINED FROM WWW
        english_lists = ["gossip", "Twitter", "politi"]
        self.warmup_epochs = warmup_epochs
        model_name = (
            "bert-base-chinese"
            if self.dataset not in english_lists
            else "bert-base-uncased"
        )
        print("BERT: using {}".format(model_name))
        self.text_model = BertModel.from_pretrained(model_name)

        # CLIP MODEL
        self.clip = (
            ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
            if self.dataset not in english_lists
            else CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        )

        self.text_attention = TokenAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.mm_attention = TokenAttention(self.unified_dim)

        for param in self.clip.parameters():
            param.requires_grad = False
        # for param in self.text_model.parameters():
        #     param.requires_grad = False
        # for param in self.image_model.parameters():
        #     param.requires_grad = False

        clip_embed_dim = 512
        self.m_i_projection = nn.Sequential(
            nn.Linear(clip_embed_dim, 256),
            nn.Linear(256, 64),
            nn.SiLU(),
        )
        self.m_t_projection = nn.Sequential(
            nn.Linear(clip_embed_dim, 256),
            nn.Linear(256, 64),
            nn.SiLU(),
        )

        # GATE, EXPERTS for features
        image_expert_list, text_expert_list, mm_expert_list = [], [], []
        for i in range(self.num_expert):
            image_expert = []
            text_expert = []
            mm_expert = []
            for j in range(self.depth):
                text_expert.append(Block(dim=self.unified_dim, num_heads=4))
                mm_expert.append(Block(dim=self.unified_dim, num_heads=4))
                image_expert.append(
                    Block(dim=self.unified_dim, num_heads=4)
                )  # note: need to output model[:,0]
            text_expert = nn.ModuleList(text_expert)
            mm_expert = nn.ModuleList(mm_expert)
            image_expert = nn.ModuleList(image_expert)
            text_expert_list.append(text_expert)
            mm_expert_list.append(mm_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)
        self.text_experts = nn.ModuleList(text_expert_list)
        self.mm_experts = nn.ModuleList(mm_expert_list)

        # self.out_unified_dim = 320
        self.image_gate_mae = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )
        self.mm_gate = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
        )

        ## MAIN TASK GATES
        self.interaction_module = InteractionModule(
            self.unified_dim, agr_threshold, sem_threshold
        )
        self.final_attention = nn.ModuleList(
            [TokenAttention(self.unified_dim) for i in range(4)]
        )
        self.fusion_SE_network_main_task = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.unified_dim, self.unified_dim),
                    nn.SiLU(),
                    nn.Linear(self.unified_dim, self.num_expert),
                )
                for i in range(4)
            ]
        )

        ## CLASSIFICATION HEAD
        self.mix_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        self.mix_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.text_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.text_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )
        self.image_trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.image_alone_classifier = nn.Sequential(
            nn.Linear(64, out_dim),
        )

        #### mapping MLPs
        self.mapping_IS_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_IS_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(1, self.unified_dim),
            nn.SiLU(),
            # nn.BatchNorm1d(self.unified_dim),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        def _expert():
            fusing_expert_ls = []
            for i in range(self.num_expert):
                fusing_expert = []
                for j in range(self.depth):
                    fusing_expert.append(Block(dim=self.unified_dim, num_heads=4))
                fusing_expert = nn.ModuleList(fusing_expert)
                fusing_expert_ls.append(fusing_expert)
            return nn.ModuleList(fusing_expert_ls)

        self.final_fusing_experts = nn.ModuleList([_expert() for i in range(2)])
        self.mm_score = None

    def get_pretrain_features(self, input_ids, attention_mask, token_type_ids, image):
        image_feature = self.image_model.forward_ying(image)
        text_feature = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

        return image_feature, text_feature

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        image,
        clip_inputs,
        image_aug=None,
        image_feature=None,
        text_feature=None,
        return_features=False,
    ):

        # print(input_ids.shape) # (24,197)
        # print(attention_mask.shape) # (24,197)
        # print(token_type_ids.shape) # (24,197)
        batch_size = image.shape[0]
        if image_aug is None:
            image_aug = image

        ## POSITIONAL ENCODINGS FOR (multimodal, text, image and modal-level)
        ## modal-level: (IMAGE TEXT MM IRRELEVANT AND VGG)
        p_1d_mm = PositionalEncoding1D(self.unified_dim)
        x_mm = torch.rand(
            batch_size, self.image_token_len + self.text_token_len, self.unified_dim
        )
        self.positional_mm = p_1d_mm(x_mm).cuda()
        p_1d_image = PositionalEncoding1D(self.unified_dim)
        x_image = torch.rand(batch_size, self.image_token_len, self.unified_dim)
        self.positional_image = p_1d_image(x_image).cuda()
        p_1d_text = PositionalEncoding1D(self.unified_dim)
        x_text = torch.rand(batch_size, self.text_token_len, self.unified_dim)
        self.positional_text = p_1d_text(x_text).cuda()
        p_1d = PositionalEncoding1D(self.unified_dim)
        x = torch.rand(batch_size, 3, self.unified_dim)
        self.positional_modal_representation = p_1d(x).cuda()

        # BASE FEATURE AND ATTENTION
        # IMAGE MAE:  OUTPUT IS (BATCH, 197, 768)
        ## FILTER OUT INVALID MODAL INFORMATION
        ## NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if image_feature is None:
            image_feature = self.image_model.forward_ying(image_aug)

        # TEXT:  INPUT IS (BATCH, WORDLEN, 768)
        ## NEWS TYPE: 0 MULTIMODAL 1 IMAGE 2 TEXT
        if text_feature is None:
            text_feature = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0]

        if self.dataset in ["weibo", "weibo21"]:
            m_i = self.clip.get_image_features(**clip_inputs)
            m_t = self.clip.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            m_i = self.clip.get_image_features(clip_inputs["pixel_values"])
            m_t = self.clip.get_text_features(
                input_ids=clip_inputs["input_ids"],
                attention_mask=clip_inputs["attention_mask"],
            )

        # print("text_feature size {}".format(text_feature.shape)) # 64,170,768
        # print("image_feature size {}".format(image_feature.shape)) # 64,197,1024
        # IMAGE ATTENTION: OUTPUT IS (BATCH, 768)
        text_atn_feature, _ = self.text_attention(text_feature)
        image_atn_feature, _ = self.image_attention(image_feature)
        mm_atn_feature, _ = self.mm_attention(
            torch.cat((image_feature, text_feature), dim=1)
        )
        gate_image_feature = self.image_gate_mae(image_atn_feature)
        gate_text_feature = self.text_gate(text_atn_feature)  # 64 320
        gate_mm_feature = self.mm_gate(mm_atn_feature)

        # IMAGE EXPERTS
        shared_image_feature = 0
        for i in range(self.num_expert):
            image_expert = self.image_experts[i]
            tmp_image_feature = image_feature
            for j in range(self.depth):
                tmp_image_feature = image_expert[j](
                    tmp_image_feature + self.positional_image
                )
            shared_image_feature += tmp_image_feature * gate_image_feature[
                :, i
            ].unsqueeze(1).unsqueeze(1)
        shared_image_feature = shared_image_feature[:, 0]

        ## TEXT
        shared_text_feature = 0
        for i in range(self.num_expert):
            text_expert = self.text_experts[i]
            tmp_text_feature = text_feature
            for j in range(self.depth):
                tmp_text_feature = text_expert[j](
                    tmp_text_feature + self.positional_text
                )  # text_feature: 64, 170, 768
            shared_text_feature += tmp_text_feature * gate_text_feature[:, i].unsqueeze(
                1
            ).unsqueeze(1)
        shared_text_feature = shared_text_feature[:, 0]

        # CONCAT TEXT-IMG
        mm_feature = torch.cat((image_feature, text_feature), dim=1)
        shared_mm_feature = 0
        for i in range(self.num_expert):
            mm_expert = self.mm_experts[i]
            tmp_mm_feature = mm_feature
            for j in range(self.depth):
                tmp_mm_feature = mm_expert[j](tmp_mm_feature + self.positional_mm)
            shared_mm_feature += tmp_mm_feature * gate_mm_feature[:, i].unsqueeze(
                1
            ).unsqueeze(1)
        shared_mm_feature = shared_mm_feature[:, 0]

        ## SCORES FOR THE FOUR MODALS
        ## NOTE: MMSCORE->0 IF IMAGE AND TEXT ARE FROM ONE NEWS
        ## AND THEREFORE SHOULD BE REVERTED AS 1-MMSCORE LATER
        ###### NOTE: HUGE MODIFICATION HAS TAKEN PLACE IN V2 ########
        ## UNIMODAL BRANCHES, NOT USED ANY MORE

        """
            shared_image_feature = self.img_projection(image_feature)
            shared_text_feature = self.text_projection(text_feature)
        """
        shared_image_feature_lite = self.image_trim(shared_image_feature)
        shared_text_feature_lite = self.text_trim(shared_text_feature)

        image_only_output = self.image_alone_classifier(shared_image_feature_lite)
        text_only_output = self.text_alone_classifier(shared_text_feature_lite)

        projected_mt = self.m_t_projection(m_t)
        projected_mi = self.m_i_projection(m_i)
        expert_mask, interaction_loss = self.interaction_module(
            text_only_output,
            image_only_output,
            shared_image_feature_lite,
            shared_text_feature_lite,
            projected_mt,
            projected_mi,
        )
        # if self.warmup_epochs > 0:
        expert_mask = torch.ones_like(expert_mask)

        ## WEIGHTED MULTIMODAL FEATURES
        is_mu = self.mapping_IS_MLP_mu(
            torch.sigmoid(image_only_output).clone().detach()
        )
        is_sigma = self.mapping_IS_MLP_sigma(
            torch.sigmoid(image_only_output).clone().detach()
        )

        t_mu = self.mapping_T_MLP_mu(torch.sigmoid(text_only_output).clone().detach())
        t_sigma = self.mapping_T_MLP_sigma(
            torch.sigmoid(text_only_output).clone().detach()
        )

        shared_image_feature = self.adaIN(
            shared_image_feature, is_mu, is_sigma
        )  # shared_image_feature * (image_atn_score)
        shared_text_feature = self.adaIN(
            shared_text_feature, t_mu, t_sigma
        )  # shared_text_feature * (text_atn_score)
        shared_mm_feature = shared_mm_feature  # shared_mm_feature #* (aux_atn_score)
        ## GATES FOR MAIN TASK
        final_fusion = []
        for k in range(4):
            concat_feature_main_biased = torch.stack(
                (
                    shared_image_feature,
                    shared_text_feature,
                    shared_mm_feature,
                ),
                dim=1,
            )
            final_feature_main_task = 0
            mask = expert_mask[:, k].unsqueeze(1)
            fusion_tempfeat_main_task, _ = self.final_attention[k](
                concat_feature_main_biased
            )
            gate_main_task = self.fusion_SE_network_main_task[k](
                fusion_tempfeat_main_task
            )
            for i in range(self.num_expert):
                fusing_expert = self.final_fusing_experts[k][i]
                tmp_fusion_feature = concat_feature_main_biased
                for j in range(self.depth):
                    tmp_fusion_feature = fusing_expert[j](
                        tmp_fusion_feature + self.positional_modal_representation
                    )
                tmp_fusion_feature = tmp_fusion_feature[:, 0]
                final_feature_main_task += tmp_fusion_feature * gate_main_task[
                    :, i
                ].unsqueeze(1)
            final_feature_main_task_lite = self.mix_trim(final_feature_main_task)
            final_fusion.append(final_feature_main_task_lite * mask)
            break

        stacked_feature = torch.stack(final_fusion, dim=0)
        fused_feature = torch.mean(stacked_feature, dim=0)
        final_output = self.mix_classifier(fused_feature)

        # NOTE: ABLATION 1: image 2. text 3. without gating 4. without regularization
        if return_features:
            return (
                image_only_output,
                image_only_output,
                text_only_output,
                # vgg_only_output,
                # aux_output,
                # torch.mean(self.irrelevant_tensor),
                (
                    final_feature_main_task_lite,
                    shared_image_feature_lite,
                    shared_text_feature_lite,
                ),
            )

        return (final_output, image_only_output, text_only_output, interaction_loss)

    def mapping(self, score):
        ## score is within 0-1
        diff_with_thresh = torch.abs(score - self.thresh)
        interval = torch.where(score - self.thresh > 0, 1 - self.thresh, self.thresh)
        return diff_with_thresh / interval


if __name__ == "__main__":
    from thop import profile

    model = Vimoe_V2()
    device = torch.device("cpu")
    input1 = torch.randn(1, 197, 768)
    input2 = torch.randn(1, 197, 768)
    flops, params = profile(model, inputs=(input1, input2))

    # stat(self.localizer.to(torch.device('cuda:0')), (3, 512, 512))
