from torch import nn
import torch
from einops import repeat

class Audio2D(nn.Module):
    def __init__(self, args, in_feat=40, LNout=512, num_patches=314, tflayers=6, pool='cls', mlp_hidden=128, num_classes=0) -> None:
        super().__init__()
        
        self.LL = nn.Linear(in_feat,LNout)
        self.lldrop = nn.Dropout(args.classify_drop)
        self.pool = pool
        PEapend = 0
        if pool == 'cls':
            PEapend = 1
            self.cls = nn.Parameter(torch.randn(1,1,LNout))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+PEapend, LNout))
        self.transdrop = args.swin_drop
        
        
        self.encoder = nn.TransformerEncoderLayer(LNout,activation="gelu",dropout=self.transdrop,nhead=8)
        self.transformer = nn.TransformerEncoder(self.encoder, tflayers)
        
        self.MLP = None
        if num_classes > 0:
            self.MLP = nn.Sequential(
                nn.LayerNorm(LNout),
                nn.Linear(LNout, mlp_hidden),
                self.lldrop,
                nn.GELU(),
                nn.Linear(mlp_hidden, num_classes)
        )
            
    def forward(self, x, mask=None):
        # x:[B,L,D]
        x = self.lldrop(self.LL(x))
        
        b,l,d = x.shape
        if self.pool == 'cls':
            cls_tokens = self.cls.repeat((b,1,1))
            x = torch.cat((cls_tokens, x), dim=1)
            l+=1
            mask=torch.concat((torch.ones(b,1).cuda(),mask),dim=1)
        x += self.pos_embedding
        # attn_mask = self.build_mask(mask)
        x = x.view(l,b,d)
        # mask = torch.transpose(mask,0,1)
        x = self.transformer(x,src_key_padding_mask=mask)
        
        if self.pool == 'cls':
            classify = x[0,:,:] # b*d
        
        if self.MLP is not None:
            prob_rate = self.MLP(classify)
            return prob_rate

        return classify
    
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