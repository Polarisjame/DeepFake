from torch import nn
import torch.nn.functional as F
from src.models.video_swin_transformer import Mlp

class Audio2D(nn.Module):
    def __init__(self, args, wav_model, in_feat=768, num_classes=2, use_feat=False) -> None:
        super().__init__()
        
        self.wav_model = wav_model
        self.use_feat = use_feat
        self.classifier = nn.Linear(512,num_classes)
        self.class_act = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, in_feat))
        self.model_drop = args.swin_drop
        if not use_feat:
            self.mlp = Mlp(in_feat, 512, 512)
            self.norm = nn.LayerNorm(512)
            self.act = nn.GELU()
            self.classify_drop = args.classify_drop
            
    def forward(self, x, mask=None):
        feat = self.wav_model(x)['last_hidden_state']
        feat = self.avgpool(feat).squeeze(1)
        feat = F.dropout(feat, self.model_drop)
        if not self.use_feat:
            classify = self.mlp(feat)
            classify = self.act(self.norm(classify))
            classify = F.dropout(classify, self.classify_drop)
            return self.class_act(self.classifier(classify).squeeze())
        return feat
    
    # def build_mask(self, mask):
    #     # mask: Tensor[B, L]
    #     b,_ = mask.shape
    #     if mask is None:
    #         return mask
    #     if self.pool == 'cls':
    #         mask=torch.concat((torch.ones(b,1).cuda(),mask),dim=1)
    #     attn_mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    #     attn_mask = attn_mask.masked_fill(attn_mask != 2, torch.inf).masked_fill(attn_mask == 2, 0)
    #     return attn_mask