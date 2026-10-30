"""Persistent, memory-mapped immutable stream audio banks.

The explicit policy versions only the derived input, not unrelated source code.
Changing build_audio_bank semantics requires bumping BANK_POLICY.
"""
from __future__ import annotations
import hashlib,json,os,tempfile,time
from pathlib import Path
import numpy as np
import torch

BANK_POLICY='stream_238_time_grid_peaks_v1'
BANK_DTYPE=np.dtype([('features','<f4',(238,)),('seconds','<f8'),('beats','<f8')])


def bank_identity(song,grid,bank_key):
    """Resolve dependency file identities without decompressing source audio."""
    from .candidates import COMPOSER_AUDIO_FEATURE_POLICY
    from .acoustics import FEATURE_POLICY
    audio=song.audio
    # Persistent training banks require an identifiable prepared audio source.
    if not hasattr(type(audio),'__getattr__') or not {'store','path'} <= vars(audio).keys():return None
    prepared=audio.store.disk_cache_path(audio.path)
    if prepared is None or not Path(prepared).is_file():return None
    def signature(path):
        path=Path(path);s=path.stat()
        return [str(path.resolve()),s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    return dict(policy=BANK_POLICY,grid=float(grid),input_key=bank_key,
        source=signature(audio.path),prepared=signature(prepared),
        composer_policy=COMPOSER_AUDIO_FEATURE_POLICY,context_policy=FEATURE_POLICY,
        feature_policy=audio.store.feature_policy,evidence_profile=audio.store.evidence_profile)


class StreamBankDiskCache:
    def __init__(self,root):
        self.root=Path(root)/'event_set_stream_banks'/BANK_POLICY
        self.hits=self.misses=self.invalid=self.write_errors=0
        self.read_sec=self.build_sec=self.write_sec=0.
        self.bytes_written=0

    def stats(self):
        return dict(path=str(self.root),hits=self.hits,misses=self.misses,invalid=self.invalid,
            write_errors=self.write_errors,read_sec=self.read_sec,build_sec=self.build_sec,
            write_sec=self.write_sec,bytes_written=self.bytes_written,load_mode='copy_on_write_mmap')

    def get(self,identity,build):
        if identity is None:return build()
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        path=self.root/(key+'.npy');tick=time.perf_counter()
        try:
            mapped=np.load(path,mmap_mode='c',allow_pickle=False)
            if mapped.ndim!=1 or mapped.dtype!=BANK_DTYPE or not len(mapped):
                raise ValueError('Invalid stream audio bank layout')
            bank=tuple(torch.from_numpy(mapped[name]) for name in ('features','seconds','beats'))
            self.hits+=1;self.read_sec+=time.perf_counter()-tick
            return bank
        except FileNotFoundError:pass
        except (ValueError,OSError,EOFError):self.invalid+=1
        self.read_sec+=time.perf_counter()-tick;self.misses+=1
        tick=time.perf_counter();bank=build();self.build_sec+=time.perf_counter()-tick
        tick=time.perf_counter();temporary=None
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            arrays=[tensor.detach().cpu().numpy() for tensor in bank]
            packed=np.empty(len(arrays[0]),dtype=BANK_DTYPE)
            for name,array in zip(('features','seconds','beats'),arrays):packed[name]=array
            fd,temporary=tempfile.mkstemp(prefix=key+'.',suffix='.tmp',dir=self.root)
            with os.fdopen(fd,'wb') as stream:np.save(stream,packed,allow_pickle=False)
            os.replace(temporary,path);temporary=None
            self.bytes_written+=path.stat().st_size
        except OSError:self.write_errors+=1
        finally:
            if temporary is not None:
                try:Path(temporary).unlink(missing_ok=True)
                except OSError:pass
            self.write_sec+=time.perf_counter()-tick
        return bank
