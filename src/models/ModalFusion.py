from torch import nn
import torch
from src.utils import Mlp
import torch.nn.functional as F
from einops import rearrange

class FusionModel(nn.Module):
    
    def __init__(self, args, VideoExtractor, AudioExtractor, PAudioExtractor, out_dim=2, video_dim=1024, audio_dim=1024, paudio_dim=768, common_dim=512):
        super(FusionModel, self).__init__()
        self.vExtract = VideoExtractor
        self.aExtract = AudioExtractor
        self.paExtract = PAudioExtractor
        self.soft=args.soft
        
        self.video_projection = nn.Linear(video_dim, common_dim)
        self.audio_projection = nn.Linear(audio_dim, common_dim)
        self.paudio_projection = nn.Linear(paudio_dim, common_dim)
        self.keys = nn.Linear(common_dim, common_dim)
        self.queries = nn.Linear(common_dim, common_dim)
        self.values = nn.Linear(common_dim, common_dim)
        self.scaling = common_dim ** -0.5
        self.attn_proj = nn.Linear(common_dim*3, 768, bias=False)
        self.norm = nn.BatchNorm1d(768, momentum=0.08)
        self.classify = Mlp(768, 256, out_dim)
        self.drop = nn.Dropout(args.classify_drop)
        # self.out_act = nn.Softmax(-1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, feature: tuple):
        video_feat, audio_feat, paudio_feat = feature
        v_x = self.vExtract(video_feat)
        a_x = self.aExtract(audio_feat)
        pa_x = self.paExtract(paudio_feat)
        
        # Attn
        v_x = self.video_projection(v_x)
        a_x = self.audio_projection(a_x)
        pa_x = self.paudio_projection(pa_x)
        
        # loss_va = self.cal_nce_loss(v_x,a_x)
        # loss_vpa = self.cal_nce_loss(v_x,pa_x)
        # loss_align = (loss_va + loss_vpa)/2
        
        comb_x = torch.cat((v_x.unsqueeze(1),a_x.unsqueeze(1),pa_x.unsqueeze(1)),dim=1) # B 3 C
        
        q = self.queries(comb_x)
        k = self.keys(comb_x)
        v = self.values(comb_x)
        energy = torch.einsum('bqd, bkd -> bqk', q, k) # B 3 3
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.drop(att)
        out = torch.einsum('bal, blv -> bav ', att, v) # B 3 C
        
        feat = rearrange(out, 'b k v -> b (k v)')
        feat = self.norm(self.attn_proj(feat))
        feat = self.drop(feat)
        feat = self.classify(feat)
        
        # feat = feat[:,1]
        # print(feat)
        
        # Linear Proj
        # v_x = self.video_projection(v_x)
        # a_x = self.audio_projection(a_x)
        # pa_x = self.paudio_projection(pa_x)
        
        # comb_x = torch.cat((v_x,a_x,pa_x),dim=-1)
        # comb_x = self.norm(self.attn_proj(comb_x))
        # comb_x = self.drop(comb_x)
        # out = self.drop(self.classify(comb_x))
        
        del v_x,a_x,pa_x,comb_x,q,k,v,att,out
        
        return self.out_act(feat.squeeze())
        
        
    def cal_nce_loss(self,p_a,p_b):
        # conduct positive and negative sample
        m1vsm2 = torch.einsum('bd,bd -> b', p_a, p_b).unsqueeze(-1)
        lse_pos = torch.logsumexp(
            m1vsm2/self.soft,
            1,
        )
        m1vsm2_all=torch.einsum('bd,cd -> bc', p_a, p_b)
        logsumexp_all_m1_vs_m2all = torch.logsumexp(
            m1vsm2_all / self.soft,
            1,
        )
        m2vsm1_all=torch.einsum('bd,cd -> bc', p_b, p_a)
        logsumexp_all_m2_vs_m1all = torch.logsumexp(
            m2vsm1_all / self.soft,
            1,
        )
        loss_m1_vs_m2 = logsumexp_all_m1_vs_m2all - lse_pos
        loss_m1_vs_m2 = torch.mean(loss_m1_vs_m2)
        loss_m2_vs_m1 = logsumexp_all_m2_vs_m1all - lse_pos
        loss_m2_vs_m1 = torch.mean(loss_m2_vs_m1)
        return loss_m1_vs_m2 + loss_m2_vs_m1
    
    
