from pathlib import Path

import torch
import torch.nn.functional as F

from gator.models.blocks import Attention, CrossAttention

class SavedAttention(Attention):
    def __init__(self, save_path: Path | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_path = save_path

    def forward(self, x, xpos, use_rope=True):

        if self._save_path is not None:
            with torch.no_grad():
                B, N, C = x.shape
                
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1,3)
                q, k, v = [qkv[:,:,i] for i in range(3)]

                qkv_save = [
                    q.cpu().detach().clone(), 
                    k.cpu().detach().clone(), 
                    v.cpu().detach().clone()
                ]
                while self._save_path.exists():
                    # increment path
                    curr_idx = self._save_path.stem.split("-")[-1]
                    new_idx = int(curr_idx) + 1
                    self._save_path = self._save_path.with_name(
                        f"{self._save_path.stem[:-len(curr_idx)]}{new_idx:03}{self._save_path.suffix}"
                    )

                print(f"Saving qkv tensors to {self._save_path}!")
                    
                torch.save(qkv_save, self._save_path)

                del q, k, v, qkv
        
        return Attention.forward(self, x, xpos, use_rope)


class SavedCrossAttention(CrossAttention):
    def __init__(self, save_path: Path | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_path = save_path

    def forward(self, query, key, value, qpos, kpos, use_rope=True):
        if self._save_path is not None:
            with torch.no_grad():
                B, Nq, C = query.shape
                Nk = key.shape[1]
                Nv = value.shape[1]
                
                q = self.projq(query).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
                k = self.projk(key).reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
                v = self.projv(value).reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)

                qkv_save = [
                    q.cpu().detach().clone(), 
                    k.cpu().detach().clone(), 
                    v.cpu().detach().clone()
                ]
                while self._save_path.exists():
                    # increment path
                    curr_idx = self._save_path.stem.split("-")[-1]
                    new_idx = int(curr_idx) + 1
                    self._save_path = self._save_path.with_name(
                        f"{self._save_path.stem[:-len(curr_idx)]}{new_idx:03}{self._save_path.suffix}"
                    )

                print(f"Saving qkv (cross-attn) tensors to {self._save_path}!")
                torch.save(qkv_save, self._save_path)

                del q, k, v
        
        return CrossAttention.forward(self, query, key, value, qpos, kpos, use_rope)
        
        
        