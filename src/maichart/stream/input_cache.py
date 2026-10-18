"""Immutable mapped training inputs, separate from the compressed source cache.

Only deterministic CPU tensors are persisted. Epoch masks are applied after
loading. Content identities and checked publication prevent stale inputs from
crossing source, clock, quality or model-input contracts.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import hashlib,json,os,pickle,tempfile,threading
from pathlib import Path

import numpy as np
import torch

SCHEMA='chart_stream_mapped_inputs_v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


class LazyBeats:
    def __init__(self,values):self.values=values
    def __len__(self):return len(self.values)
    def __getitem__(self,index):
        value=self.values[index]
        return tuple(map(Fraction,value)) if isinstance(index,slice) else Fraction(value)
    def __iter__(self):return map(Fraction,self.values)


class MappedInputCache:
    def __init__(self,cache_dir):
        self.root=Path(cache_dir)/'stream_training_inputs'/SCHEMA
        try:self.root.mkdir(parents=True,exist_ok=True);self.writable=True
        except OSError:self.writable=False
        from maichart.form.cache_support import source_digests
        self.digests=source_digests(str(self.root))
        self._locks=[threading.Lock() for _ in range(64)]
        self._guard=threading.Lock()
        self.counts=dict(hits=0,rebuilt=0,invalid=0,write_errors=0)
        self._validated=OrderedDict()

    def _count(self,key):
        with self._guard:self.counts[key]+=1

    def get(self,identity,builder):
        key=fingerprint({'schema':SCHEMA,'identity':identity})
        index=self.root/(key+'.json')
        with self._locks[int(key[:8],16)%len(self._locks)]:
            try:
                info=json.loads(index.read_text(encoding='utf8'))
                name=info['file']
                if Path(name).name!=name or not name.startswith(key+'.'):raise ValueError('Invalid cache file')
                path=self.root/name
                stat=path.stat();signature=(name,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
                with self._guard:validated=signature in self._validated
                if not validated and self.digests.digest(path)!=info['sha256']:
                    raise ValueError('Mapped training input checksum mismatch')
                saved=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
                if saved.get('schema')!=SCHEMA or saved.get('key')!=key:
                    raise ValueError('Mapped training input identity mismatch')
                if not validated:
                    for value in saved['tensors'].values():
                        if not torch.is_tensor(value) or value.device.type!='cpu':raise ValueError('Invalid cached tensor')
                        if value.is_floating_point() and not np.isfinite(value.numpy()).all():raise ValueError('Nonfinite cached tensor')
                    with self._guard:
                        self._validated[signature]=True
                        while len(self._validated)>4096:self._validated.popitem(last=False)
                self._count('hits');return saved['tensors'],saved['extra']
            except FileNotFoundError:pass
            except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError):
                self._count('invalid')
            tensors,extra=builder()
            if not self.writable:
                self._count('write_errors');return tensors,extra
            # A view can otherwise serialize its entire parent storage.
            owned={name:value.detach().cpu().contiguous().clone() for name,value in tensors.items()}
            path=temporary=None
            try:
                fd,name=tempfile.mkstemp(prefix=key+'.',suffix='.pt',dir=self.root);os.close(fd)
                path=Path(name)
                torch.save({'schema':SCHEMA,'key':key,'tensors':owned,'extra':extra},path)
                digest=self.digests.digest(path)
                with tempfile.NamedTemporaryFile(dir=self.root,suffix='.tmp',delete=False,mode='w',encoding='utf8') as f:
                    temporary=Path(f.name);json.dump({'file':path.name,'sha256':digest},f)
                temporary.replace(index)
                self._count('rebuilt')
            except (OSError,RuntimeError):
                self._count('write_errors')
                if temporary is not None:temporary.unlink(missing_ok=True)
                if path is not None:path.unlink(missing_ok=True)
            return tensors,extra

    def stats(self):
        with self._guard:return {**self.counts,'path':str(self.root)}


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    return 0
