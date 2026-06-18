from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

# 修正前
# from modules.F0Predictor.F0Predictor import F0Predictor

# 修正後（直接自分自身をインポートするか、ベースクラスを無視する設定にします）
# ※ F0Predictor.py も同じフォルダにある場合は以下で通ります
try:
    from F0Predictor import F0Predictor
except ImportError:
    # ベースクラスが見つからない場合、最小限の定義で代用
    class F0Predictor:
        def __init__(self, *args, **kwargs): pass
        def compute_f0(self, *args, **kwargs): pass

from rmvpe import RMVPE


class RMVPEF0Predictor(F0Predictor):
# 18行目付近：定義部分
    # 引数の名前を 'model_path' に固定します
    # 必要な材料（hop_length, f0_min, f0_max など）をすべて入り口で受け取れるようにします。
    # hop_length はトレーニング時に合わせた 320 をデフォルト値に設定しました。
    def __init__(self, model_path, device, dtype=torch.float32, hop_length=160, f0_min=50, f0_max=1100, threshold=0.03, sampling_rate=16000):
        # 1. デバイスの確定
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.dtype = dtype
        self.model_path = model_path
        
        # 2. RMVPE本体のロード
        # 外から受け取った model_path を正しく渡します
        self.rmvpe = RMVPE(model_path=model_path, dtype=dtype, device=self.device)
        
        # 3. 各種パラメータを「self（自分自身の箱）」に保存
        # こうすることで、他の関数（compute_f0など）でもこれらの値が使えるようになります
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.name = "rmvpe"

    def repeat_expand(
        self, content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
    ):
        ndim = content.ndim

        if content.ndim == 1:
            content = content[None, None]
        elif content.ndim == 2:
            content = content[None]

        assert content.ndim == 3

        is_np = isinstance(content, np.ndarray)
        if is_np:
            content = torch.from_numpy(content)

        results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

        if is_np:
            results = results.numpy()

        if ndim == 1:
            return results[0, 0]
        elif ndim == 2:
            return results[0]

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = self.repeat_expand(f0, pad_to)
        
        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        
        # 去掉0频率, 并线性插值
        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        
        vuv_vector = F.interpolate(vuv_vector[None,None,:],size=pad_to)[0][0]

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device).cpu().numpy(),vuv_vector.cpu().numpy()
        if f0.shape[0] == 1:
            return (torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]).cpu().numpy() ,vuv_vector.cpu().numpy()
    
        # 大概可以用 torch 重写?
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        #vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector,pad_to/len(vuv_vector),order = 0))
        
        return f0,vuv_vector.cpu().numpy()

    def compute_f0(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)[0]
    
    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)