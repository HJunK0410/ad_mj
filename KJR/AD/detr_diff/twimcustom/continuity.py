import torch
import torch.nn as nn
import math

import torch.nn.functional as F

# -------------------------------------
# 2d 연속성...
# -------------------------------------

class MaskedConv2d(nn.Conv2d):
    #2D causal conv: 현재 위치보다 위쪽/왼쪽 픽셀만 참조
    #PixelCNN 방식의 mask 적용
    def __init__(self, in_ch, out_ch, kernel_size, mask_type='B', padding_mode='reflect', **kwargs):
        super().__init__(in_ch, out_ch, kernel_size, **kwargs)
        assert mask_type in ['A', 'B']
        assert padding_mode in ['reflect', 'replicate']
        self.mask_type = mask_type
        self.padding_mode = padding_mode
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, h, w = self.weight.size()
        yc, xc = h // 2, w // 2

        # mask 설정: 아래쪽 제거, 현재 행에서는 오른쪽 제거
        self.mask[:, :, yc+1:, :] = 0
        self.mask[:, :, yc, xc + (mask_type == 'A'):] = 0

    def forward(self, x, context_mask=None):
        """
        x: [B, C, H, W] 입력
        context_mask: 사용하지 않음 (호환성을 위해 유지)
        """
        # 가중치에 mask 적용 (인과적 conv)
        # 가중치가 죽는버그...20251118
        #self.weight.data *= self.mask
        #return super().forward(x)
        w = self.weight * self.mask
        
        # 경계 halo 해결: reflect/replicate 패딩으로 변경
        pad = self.padding
        if isinstance(pad, int):
            pad_h = pad_w = pad
        elif isinstance(pad, (tuple, list)):
            if len(pad) == 2:
                pad_h, pad_w = pad
            else:
                pad_h, pad_w = pad[0], pad[1]
        else:
            pad_h = pad_w = 0
        
        # reflect/replicate 패딩 적용
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=self.padding_mode)
        else:
            x_padded = x
        
        # 기존 동작 (하지만 padding은 reflect/replicate)
        return F.conv2d(x_padded, w, self.bias,
                       self.stride, 0,  # padding=0 (이미 패딩됨)
                       self.dilation, self.groups)

class ContinuityHead2D(nn.Module):
    """
    2D causal conv 기반 연속성 예측기
    입력: [B, C, H, W]
    출력: z, z_hat, (H, W)
    """
    def __init__(self, in_ch, d=256, k=3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d, 1)  # 1차원 flatt하지 않을것...
        # Sequential을 분리하여 MaskedConv2d에 마스크를 전달할 수 있도록 함
        self.masked_conv = MaskedConv2d(d, d, k, padding=k//2, mask_type='B')
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(d, d, 1)

    def forward(self, x, norm_mask=None):
        """
        x: [B, C, H, W]
        norm_mask: 사용하지 않음 (호환성을 위해 유지)
        z: [B, D, H, W] — encoder feature (projected)
        z_hat: [B, D, H, W] — causal conv2d 예측 결과
        """
        z = self.proj(x)          # [B, D, H, W]
        
        # MaskedConv2d 호출 (context_mask는 사용하지 않음)
        z_hat = self.masked_conv(z)
        z_hat = self.relu(z_hat)
        z_hat = self.conv1x1(z_hat)
        
        return z, z_hat, (z.shape[2], z.shape[3])

def npp_loss_2d(z, z_hat, norm_mask=None, mode='l1'):
    # z, z_hat: [B, D, H, W]
    # norm_mask: [B, 1, H, W]
    if mode == 'l1':
        diff = (z - z_hat).abs().mean(1, keepdim=True)  # [B,1,H,W]
    elif mode == 'cos':
        #a = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        #b = z_hat / (z_hat.norm(dim=1, keepdim=True) + 1e-6)
        #diff = (1.0 - (a * b).sum(1, keepdim=True))  # [B,1,H,W]
        a = z / (z.norm(dim=1, keepdim=True).clamp(min=1e-6))
        b = z_hat / (z_hat.norm(dim=1, keepdim=True).clamp(min=1e-6))
        diff = (1.0 - (a * b).sum(1, keepdim=True).clamp(-1.0,1.0))  # [B,1,H,W]
    else:
        raise ValueError("mode must be 'l1' or 'cos'")

    # 마스크 적용
    if norm_mask is not None:
        if norm_mask.shape[1] == 1:
            w = norm_mask
        else:
            w = norm_mask.unsqueeze(1)  # [B,1,H,W] 보장
        loss = (diff * w).sum() / (w.sum() + 1e-6)
    else:
        loss = diff.mean()
    return loss
    # [B, D, H, W] → [B, T, D]
    #B, D, H, W = z.shape
    #z = z.permute(0, 2, 3, 1).reshape(B, H*W, D)
    #z_hat = z_hat.permute(0, 2, 3, 1).reshape(B, H*W, D)

    #return npp_loss(z, z_hat, norm_mask, mode)

############################################################


class SeqProjector(nn.Module):
    # CxHxW -> DxT (D는 임베딩 축)
    def __init__(self, in_ch, d=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d, 1)

    def forward(self, x): # x: [B,C,H,W]
        x = self.proj(x)         # [B,D,H,W]
        B, D, H, W = x.shape
        x = x.view(B, D, H*W)    # [B,D,T]
        x = x.transpose(1, 2)    # [B,T,D]
        return x, (H, W)

class CausalConv1D(nn.Module):
    # 가벼운 causal conv 예측기
    def __init__(self, d=256, k=3, hidden=256):
        super().__init__()
        pad = k-1
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, k, padding=pad),  # causal-like: 나중에 crop
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, d, 1)
        )
        self.k = k

    def forward(self, z):  # z: [B,T,D]
        zt = z.transpose(1, 2)  # [B,D,T]
        y = self.net(zt)        # [B,D,T+pad]
        y = y[:, :, :zt.size(2)]  # crop to causal
        y = y.transpose(1, 2)   # [B,T,D]
        # 예측은 z_t를 z_{<t}로: 첫 토큰은 예측 불가 → 쉬프트 비교에서 제외
        return y

class ContinuityHead(nn.Module):
    def __init__(self, in_ch, d=256, k=3):
        super().__init__()
        self.proj = SeqProjector(in_ch, d)
        self.pred = CausalConv1D(d=d, k=k)

    def forward(self, x):  # x: [B,C,H,W]
        z, hw = self.proj(x)         # [B,T,D], (H,W)
        z_hat = self.pred(z)         # [B,T,D]
        return z, z_hat, hw

def next_pred_loss(z, z_hat, w=None, mode="l1"):
    """다음 예측 손실: z_hat과 z의 차이"""
    if z.dim() != 3 or z_hat.dim() != 3:
        raise ValueError(f"z/z_hat must be [B,T,D], got {tuple(z.shape)} {tuple(z_hat.shape)}")
    if z.shape != z_hat.shape:
        raise ValueError(f"Shape mismatch: z={tuple(z.shape)} vs z_hat={tuple(z_hat.shape)}")

    if mode == "l1":
        per = (z_hat - z).abs().mean(-1)
    elif mode == "l2":
        per = ((z_hat - z) ** 2).mean(-1).sqrt()
    else:
        raise ValueError("mode must be 'l1' or 'l2'")

    if w is None:
        return per.mean()

    w = align_mask_to_T(w, per.shape[1])
    return (per * w).sum() / (w.sum() + 1e-6)


def npp_loss(z, z_hat, norm_mask=None, mode='l1', loss_type="next+diff", next_w=1.0, diff_w=1.0, alpha=0.3, beta=1.0):
    """
    Next Prediction + Differencing Loss
    
    Args:
        z: [B,T,D] 실제 임베딩
        z_hat: [B,T,D] 예측 임베딩
        norm_mask: [B,T] 마스크
        mode: "l1" or "l2"
        loss_type: "next", "diff", or "next+diff"
        next_w: next_pred_loss 가중치
        diff_w: diff_loss 가중치
        alpha: 1차 차분 가중치
        beta: 2차 차분 가중치
    """
    # z, z_hat: [B,T,D]
    # norm_mask: [B,T] (정상=1, 결함/경계=0~0.5). None이면 전부 1
    
    l_next = next_pred_loss(z, z_hat, w=norm_mask, mode=mode)
    
    if loss_type == "next":
        return l_next
    if loss_type == "diff":
        return diff_loss_1d(z, norm_mask=norm_mask, alpha=alpha, beta=beta, mode=mode) if diff_w > 0 else z.new_tensor(0.0)
    
    # loss_type == "next+diff"
    l_diff = diff_loss_1d(z, norm_mask=norm_mask, alpha=alpha, beta=beta, mode=mode) if diff_w > 0 else z.new_tensor(0.0)
    return next_w * l_next + diff_w * l_diff

def delta_map(z, z_hat, hw):
    # Δ = |z - z_hat| 평균 → HxW로 복원
    B, T, D = z.size()
    H, W = hw
    d = (z[:,1:,:] - z_hat[:,1:,:]).abs().mean(-1)  # [B,T-1]
    pad = torch.zeros(B,1, device=z.device, dtype=d.dtype)
    d = torch.cat([pad, d], dim=1)                   # 첫 토큰 0 padding
    d = d.view(B, H, W)                              # [B,H,W]
    return d


# -------------------------------------
# 차분 로스 (Differencing Loss)
# -------------------------------------

def _mask_to_BT(w):
    """마스크를 [B,T] 형태로 변환"""
    if w is None:
        return None
    if not torch.is_tensor(w):
        raise TypeError(f"w must be torch.Tensor or None, got {type(w)}")
    
    if w.dim() == 2:
        return w
    if w.dim() == 3:
        B, H, W = w.shape
        return w.view(B, H * W)
    if w.dim() == 4:
        if w.size(1) != 1:
            w = w.mean(1, keepdim=True)
        B, _, H, W = w.shape
        return w.view(B, H * W)
    
    raise ValueError(f"w must be [B,T] or spatial mask, got {tuple(w.shape)}")


def align_mask_to_T(w, target_T):
    """마스크를 target_T 길이에 맞춤"""
    if w is None:
        return None
    w = _mask_to_BT(w)
    B, Tw = w.shape
    
    if Tw == target_T:
        return w.contiguous()
    if Tw > target_T:
        return w[:, :target_T].contiguous()
    
    pad = w.new_zeros((B, target_T - Tw))
    return torch.cat([w, pad], dim=1).contiguous()


def diff_loss_1d(z, norm_mask=None, alpha=0.3, beta=1.0, mode="l1"):
    """
    1차 차분(Δz): 인접 패치 임베딩 변화량을 제약하여 급격한 변동을 억제 (smoothness of variation)
    2차 차분(Δ²z): 변화량의 변화(곡률)를 제약하여 임베딩 궤적의 급격한 꺾임을 억제 (curvature regularization)

    최종 손실:  alpha * L(Δz) + beta * L(Δ²z)
    
    Args:
        z: [B,T,D] 형태의 임베딩 시퀀스
        norm_mask: [B,T] 또는 [B,H,W] 형태의 마스크 (정상=1, 결함/경계=0~0.5)
        alpha: 1차 차분 가중치 (기본값: 0.3)
        beta: 2차 차분 가중치 (기본값: 1.0)
        mode: "l1" 또는 "l2" (기본값: "l1")
    """
    if z.dim() != 3:
        raise ValueError(f"z must be [B,T,D], got {tuple(z.shape)}")
    B, T, D = z.shape
    if T < 3:
        return z.new_tensor(0.0)
    
    # 1차 차분: v_t = z_t - z_{t-1}
    v = z[:, 1:, :] - z[:, :-1, :]  # [B,T-1,D]
    
    # 2차 차분: a_t = v_t - v_{t-1} = z_t - 2 z_{t-1} + z_{t-2}
    a = v[:, 1:, :] - v[:, :-1, :]  # [B,T-2,D]
    
    if mode == "l1":
        d1 = v.abs().mean(-1)       # [B,T-1]
        d2 = a.abs().mean(-1)       # [B,T-2]
    elif mode == "l2":
        d1 = (v ** 2).mean(-1).sqrt()
        d2 = (a ** 2).mean(-1).sqrt()
    else:
        raise ValueError("mode must be 'l1' or 'l2'")
    
    if norm_mask is None:
        return float(alpha) * d1.mean() + float(beta) * d2.mean()
    
    nm = align_mask_to_T(norm_mask, T)  # [B,T]
    w1 = nm[:, 1:]                      # [B,T-1] (Δz 가중치)
    w2 = nm[:, 2:]                      # [B,T-2] (Δ²z 가중치)
    
    l1 = (d1 * w1).sum() / (w1.sum() + 1e-6)
    l2 = (d2 * w2).sum() / (w2.sum() + 1e-6)
    return float(alpha) * l1 + float(beta) * l2