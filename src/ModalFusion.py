from torch import nn
import torch
from src.video_swin_transformer import Mlp

class FusionModel(nn.Module):
    
    def __init__(self, args, VideoExtractor, AudioExtractor, PAudioExtractor, video_dim=1024, audio_dim=1024, paudio_dim=768, common_dim=512):
        super(FusionModel, self).__init__()
        self.vExtract = VideoExtractor
        self.aExtract = AudioExtractor
        self.paExtract = PAudioExtractor
        self.video_projection = nn.Linear(video_dim, common_dim)
        self.audio_projection = nn.Linear(audio_dim, common_dim)
        self.paudio_projection = nn.Linear(paudio_dim, common_dim)
        self.attn_proj = nn.Linear(common_dim*3, 768, bias=False)
        self.norm = nn.BatchNorm1d(768)
        self.classify = Mlp(768, 256, 2)
        self.drop = nn.Dropout(args.classify_drop)
        self.out_act = nn.Softmax(-1)
        # self.out_act = nn.Sigmoid()
        
    def forward(self, feature: tuple):
        video_feat, audio_feat, paudio_feat = feature
        v_x = self.vExtract(video_feat)
        a_x = self.aExtract(audio_feat)
        pa_x = self.paExtract(paudio_feat)
        v_x = self.video_projection(v_x)
        a_x = self.audio_projection(a_x)
        pa_x = self.paudio_projection(pa_x)
        
        comb_x = torch.cat((v_x,a_x,pa_x),dim=-1)
        comb_x = self.norm(self.attn_proj(comb_x))
        comb_x = self.drop(comb_x)
        out = self.drop(self.classify(comb_x))
        
        del v_x,a_x,pa_x,comb_x
        
        return self.out_act(out.squeeze())
        
        
    # def cal_nce_loss(self,video_proj,audio_proj):
    #     # conduct positive and negative sample
    #     b, _ = video_proj.shape
    #     m1vsm2 = torch.einsum('bmd,bnd -> bbmn', video_proj, audio_proj)
    #     m1_vs_m2 = m1vsm2.view(b,b,-1)
    #     sim_pos = torch.einsum("bbn->bn", m1_vs_m2) 
    #     lse_pos = torch.logsumexp(
    #         sim_pos/self.soft_param,
    #         1,
    #     )
    #     m1vsm2_all = torch.einsum('bmd,bnd -> bbmn', video_proj, audio_proj)
    #     m1_vs_m2_all = m1vsm2_all.view(b,-1)
    #     logsumexp_all_m1_vs_m2all = torch.logsumexp(
    #     m1_vs_m2_all / self.soft_param,
    #     1,
    #     )
    #     m2vsm1_all = torch.einsum('bmd,bnd -> bbmn', audio_proj, video_proj)
    #     m2_vs_m1_all = m2vsm1_all.view(b,-1)
    #     logsumexp_all_m2_vs_m1all = torch.logsumexp(
    #     m2_vs_m1_all / self.soft_param,
    #     1,
    #     )
    #     loss_m1_vs_m2 = logsumexp_all_m1_vs_m2all - lse_pos
    #     loss_m1_vs_m2 = torch.mean(loss_m1_vs_m2)
    #     loss_m2_vs_m1 = logsumexp_all_m2_vs_m1all - lse_pos
    #     loss_m2_vs_m1 = torch.mean(loss_m2_vs_m1)
    #     return loss_m1_vs_m2 + loss_m2_vs_m1
        