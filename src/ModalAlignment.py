from torch import nn
import torch

class VAModel(nn.Module):
    
    def __init__(self, args, VideoExtractor, AudioExtractor, video_dim=512, audio_dim=1024, common_dim=512):
        super(VAModel, self).__init__()
        self.vExtract = VideoExtractor
        self.aExtract = AudioExtractor
        self.video_projection = nn.Linear(video_dim, common_dim)
        self.audio_projection = nn.Linear(audio_dim, common_dim)
        self.soft_param = args.soft_param
        
    def forward(self, x):
        _, raw_audio_feat = self.aExtract(x)
        _, raw_vedio_feat = self.vExtract(x)
        audio_proj = self.audio_projection(raw_audio_feat)
        video_proj = self.video_projection(raw_vedio_feat)
        return self.cal_nce_loss(video_proj, audio_proj)
        
    def cal_nce_loss(self,video_proj,audio_proj):
        # conduct positive and negative sample
        b, _ = video_proj.shape
        m1vsm2 = torch.einsum('bmd,bnd -> bbmn', video_proj, audio_proj)
        m1_vs_m2 = m1vsm2.view(b,b,-1)
        sim_pos = torch.einsum("bbn->bn", m1_vs_m2) 
        lse_pos = torch.logsumexp(
            sim_pos/self.soft_param,
            1,
        )
        m1vsm2_all = torch.einsum('bmd,bnd -> bbmn', video_proj, audio_proj)
        m1_vs_m2_all = m1vsm2_all.view(b,-1)
        logsumexp_all_m1_vs_m2all = torch.logsumexp(
        m1_vs_m2_all / self.soft_param,
        1,
        )
        m2vsm1_all = torch.einsum('bmd,bnd -> bbmn', audio_proj, video_proj)
        m2_vs_m1_all = m2vsm1_all.view(b,-1)
        logsumexp_all_m2_vs_m1all = torch.logsumexp(
        m2_vs_m1_all / self.soft_param,
        1,
        )
        loss_m1_vs_m2 = logsumexp_all_m1_vs_m2all - lse_pos
        loss_m1_vs_m2 = torch.mean(loss_m1_vs_m2)
        loss_m2_vs_m1 = logsumexp_all_m2_vs_m1all - lse_pos
        loss_m2_vs_m1 = torch.mean(loss_m2_vs_m1)
        return loss_m1_vs_m2 + loss_m2_vs_m1
        