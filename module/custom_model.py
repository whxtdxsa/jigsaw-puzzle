import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, mask_ratio = 0.0, pretrained = True):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.pretrained = pretrained

        # Use a smaller variant of DEiT3
        deit3 = timm.create_model('deit3_base_patch16_384', pretrained = pretrained)
        # deit3 = timm.create_model('deit3_small_patch16_224', pretrained=pretrained)

        # Use relevant parts of DEiT3
        self.patch_embed = deit3.patch_embed
        self.cls_token = deit3.cls_token
        self.blocks = deit3.blocks
        self.norm = deit3.norm

        # Simplified jigsaw head
        self.jigsaw = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 24*24)
        )
        # self.jigsaw = nn.Linear(384, 24*24)  # Adjust number of neurons

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # The length which not be masked

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        # target = einops.repeat(self.target, 'L -> N L', N=N)
        # target = target.to(x.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep
        return x_masked, target_masked

    def forward(self, x):
        x = self.patch_embed(x)
        x, target = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, 24*24), target.reshape(-1)

    def forward_test(self, x):
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x