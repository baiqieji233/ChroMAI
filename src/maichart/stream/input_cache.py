"""Immutable mapped training inputs, separate from the compressed source cache.

Only deterministic CPU tensors are persisted. Epoch masks are applied after
loading. Content identities and checked publication prevent stale inputs from
crossing source, clock, quality or model-input contracts.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import hashlib,json,os,pickle,tempfile,threading,zlib
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


class CompactBeatText:
    """Preserve exact spelling without retaining millions of Python strings."""
    def __init__(self,values):
        self.count=len(values)
        self.data=zlib.compress('\n'.join(values).encode('ascii'),1)
    def __len__(self):return self.count
    def _values(self):return zlib.decompress(self.data).decode('ascii').splitlines()
    def __iter__(self):return iter(self._values())
    def __getitem__(self,index):return self._values()[index]


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
        self._validated=OrderedDict();self.validation_capacity=32768
        self._resident={};self.resident_bytes=0;self.resident_budget=0
        self.resident_hits=0;self.file_reads=0;self.validation_bytes=0

    @staticmethod
    def key(identity):return fingerprint({'schema':SCHEMA,'identity':identity})

    def _read(self,key):
        index=self.root/(key+'.json')
        info=json.loads(index.read_text(encoding='utf8'))
        name=info['file']
        if Path(name).name!=name or not name.startswith(key+'.'):raise ValueError('Invalid cache file')
        path=self.root/name
        stat=path.stat();signature=(name,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        with self._guard:validated=signature in self._validated;self.file_reads+=1
        if not validated and self.digests.digest(path)!=info['sha256']:
            raise ValueError('Mapped training input checksum mismatch')
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        if saved.get('schema')!=SCHEMA or saved.get('key')!=key:
            raise ValueError('Mapped training input identity mismatch')
        if not isinstance(saved.get('tensors'),dict) or not isinstance(saved.get('extra'),dict):
            raise ValueError('Invalid mapped training input payload')
        if not validated:
            scanned=0
            for value in saved['tensors'].values():
                if not torch.is_tensor(value) or value.device.type!='cpu':raise ValueError('Invalid cached tensor')
                if value.is_floating_point():
                    scanned+=value.numel()*value.element_size()
                    if not np.isfinite(value.numpy()).all():raise ValueError('Nonfinite cached tensor')
            with self._guard:
                self.validation_bytes+=scanned;self._validated[signature]=True
                # Covers the complete active dataset; metadata only, no tensors.
                while len(self._validated)>self.validation_capacity:self._validated.popitem(last=False)
        return saved['tensors'],saved['extra']

    def preload(self,identities,*,budget_bytes,policy='auto',report=lambda **_:None,memory=None,
                reserve_bytes=0,validation_records=0):
        """Clone trusted file mappings into bounded anonymous CPU storage.

        Exact identities are supplied by the active dataset, in query-first
        priority. Auto keeps mapped fallback when capacity or RAM is low.
        Resident explicitly requires the complete existing input set to fit.
        """
        if policy not in {'auto','mapped','resident'}:raise ValueError('Invalid residency policy')
        from maichart.form.audio_residency import physical_memory
        memory=memory or physical_memory
        unique=list(dict.fromkeys(self.key(identity) for identity in identities))
        self.validation_capacity=int(validation_records) or max(1,len(unique))
        if policy=='mapped':return {'mode':'mapped','loaded':0,'budget_bytes':budget_bytes,
            'validation_records':self.validation_capacity}
        self.resident_budget=int(budget_bytes)
        missing=capacity_skips=memory_skips=0
        initial=memory();reserve=int(reserve_bytes) or (max(2*1024**3,int(initial[0]*.1)) if initial else 2*1024**3)
        if self.resident_budget<=0:
            if policy=='resident':raise ValueError('全部驻留需要正的驻留内存预算')
            return {'mode':'mapped','loaded':0,'total':len(unique),'bytes':0,'budget_bytes':0,
                'memory_reserve_bytes':reserve,'validation_records':self.validation_capacity}
        report(stage='residency',completed=0,total=len(unique),bytes=0,budget_bytes=self.resident_budget)
        for i,key in enumerate(unique,1):
            if self.resident_bytes>=self.resident_budget:
                capacity_skips+=1
            elif (available:=memory()) and available[1]<=reserve:
                memory_skips+=1
            else:
                try:tensors,extra=self._read(key)
                except FileNotFoundError:missing+=1
                except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError):
                    self._count('invalid');missing+=1
                else:
                    raw_beats=extra.get('beats')
                    compact_beats=CompactBeatText(raw_beats) if raw_beats else None
                    size=tensor_bytes(tensors)+1024+128*len(tensors)+(len(compact_beats.data) if compact_beats is not None else 0)
                    if self.resident_bytes+size>self.resident_budget:
                        capacity_skips+=1
                    elif available and available[1]<size+reserve:
                        memory_skips+=1
                    else:
                        extra=dict(extra)
                        if compact_beats is not None:extra['beats']=compact_beats
                        # clone() gives ordinary heap storage; retaining an mmap alone
                        # cannot prevent the OS from fetching its pages from disk again.
                        owned={name:value.clone(memory_format=torch.contiguous_format) for name,value in tensors.items()}
                        with self._guard:self._resident[key]=(owned,extra);self.resident_bytes+=size
            if i%100==0 or i==len(unique):
                report(stage='residency',completed=i,total=len(unique),loaded=len(self._resident),
                    bytes=self.resident_bytes,budget_bytes=self.resident_budget)
        mode='resident' if len(self._resident)==len(unique) else 'partial_resident'
        result={'mode':mode,'loaded':len(self._resident),'total':len(unique),'bytes':self.resident_bytes,
            'budget_bytes':self.resident_budget,'missing':missing,'capacity_skips':capacity_skips,
            'memory_skips':memory_skips,'memory_reserve_bytes':reserve,
            'validation_records':self.validation_capacity}
        report(stage='residency',completed=len(unique),**result)
        if policy=='resident' and mode!='resident':
            raise ValueError('全部输入无法驻留：请提高缓存内存预算或选择自动驻留。'+str(result))
        return result

    def _count(self,key):
        with self._guard:self.counts[key]+=1

    def validate_existing(self,identity):
        """Preflight persisted inputs with the same checks as a real read.

        Do not build epoch-dependent dataset samples just to discover that
        their immutable inputs were already prepared. Failed validation is
        left to get()'s normal rebuild path, preserving its error accounting.
        """
        key=self.key(identity)
        with self._locks[int(key[:8],16)%len(self._locks)]:
            with self._guard:
                if key in self._resident:return True
            try:self._read(key)
            except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,pickle.UnpicklingError):return False
        return True

    def get(self,identity,builder):
        key=self.key(identity)
        index=self.root/(key+'.json')
        with self._locks[int(key[:8],16)%len(self._locks)]:
            with self._guard:
                resident=self._resident.get(key)
                if resident is not None:
                    self.resident_hits+=1;self.counts['hits']+=1;return resident
            try:
                tensors,extra=self._read(key)
                self._count('hits');return tensors,extra
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
        with self._guard:return {**self.counts,'path':str(self.root),'file_reads':self.file_reads,
            'validation_bytes':self.validation_bytes,'resident_hits':self.resident_hits,
            'resident_entries':len(self._resident),'resident_bytes':self.resident_bytes,
            'resident_budget_bytes':self.resident_budget,'validation_records':self.validation_capacity,
            'source_fingerprints':self.digests.stats()}


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    return 0
