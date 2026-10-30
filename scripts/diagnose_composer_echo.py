"""Step-zero teacher-forced echo diagnosis; existing caches only, no training."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, os, threading, time
from pathlib import Path


def readonly_caches(cache_dir):
    from maichart.form import cache_support, discovery_index, preparation_cache, data as form_data
    from maichart.composer import data, structure
    class MemoryDigests:
        def __init__(self): self.values = {}; self.lock = threading.RLock()
        def digest(self, path):
            path = Path(path).resolve(); st = path.stat(); key = (str(path),st.st_size,st.st_mtime_ns)
            with self.lock:
                if key not in self.values:
                    with path.open('rb') as stream:self.values[key] = hashlib.file_digest(stream,'sha256').hexdigest()
                return self.values[key]
        def stats(self): return dict(readonly=True,files=len(self.values))
    digests = MemoryDigests()
    cache_support.source_digests = lambda *_:digests
    discovery_index.DiscoveryIndex.__init__ = lambda self,*_:setattr(self,'connection',None)
    preparation_cache.PreparationCache._promote_legacy = lambda *a,**k:None
    original = preparation_cache.PreparationCache.load
    def load_existing(self,*args,**kwargs):
        result,status = original(self,*args,**kwargs)
        if result is None:raise RuntimeError(f'Read-only diagnosis: preparation cache unavailable ({status})')
        return result,status
    preparation_cache.PreparationCache.load = load_existing
    def forbidden(*a,**kw):raise RuntimeError('Read-only diagnosis refuses cache reconstruction or publication')
    preparation_cache.PreparationCache.save = forbidden
    form_data._AudioStore._load_source = forbidden
    data.composer_audio_nodes = forbidden
    data.publish_bank = forbidden
    data.ComposerDataset._save_bounds_cache = staticmethod(lambda *a,**kw:None)
    structure._build = forbidden
    structure._publish_structure_array = forbidden
    roots = tuple(str(p.resolve()).casefold()+os.sep for p in cache_support.cache_roots(str(cache_dir)))
    def protected(path):
        return isinstance(path,(str,bytes,os.PathLike)) and str(Path(os.fsdecode(path)).resolve()).casefold().startswith(roots)
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if protected(path) and ((isinstance(mode,str) and any(x in mode for x in 'wax+')) or
                                   isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
                raise RuntimeError(f'Read-only cache write blocked: {path}')
        if event in ('os.remove','os.rename','os.mkdir','os.link','os.symlink') and any(protected(p) for p in args[:2]):
            # mkdir(exist_ok=True) is still a mutation attempt; diagnose it.
            raise RuntimeError(f'Read-only cache mutation blocked: {event} {args[:2]}')
    sys.addaudithook(audit)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='checkpoints/prism_composer_v23/composer_inference.pt')
    p.add_argument('--manifest',default='checkpoints/prism_composer_v22/data_split/dev_manifest.json')
    p.add_argument('--cache-dir',default='cache/chart_composer_v1')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--max-batches',type=int,default=0)
    a=p.parse_args()
    import torch
    from maichart.composer.checkpoint import load
    from maichart.composer.data import ComposerDataset
    from maichart.composer.evaluation import evaluate_teacher_forced
    from maichart.composer.objectives import action_group_loss_kwargs
    torch.set_num_threads(2)
    model,payload=load(a.checkpoint,device=a.device)
    c=model.config
    readonly_caches(a.cache_dir)
    started=time.perf_counter()
    ds=ComposerDataset(a.manifest,cache_dir=a.cache_dir,candidate_count=c.candidate_count,
        history_events=c.history_events,action_slots=c.action_slots,architecture_profile=c.architecture_profile,
        action_state_policy=c.action_state_policy,head_timing_policy=c.head_timing_policy,
        motif_memory_policy=c.motif_memory_policy,motif_max_events=c.motif_max_events,
        structure_policy=c.structure_policy,structure_backend=c.structure_backend,
        atom_catalog=payload['atom_catalog'],quality_inputs_enabled=c.configuration_quality_policy!='off',
        quality_inputs_policy=c.configuration_quality_policy,preparation_workers=1,audio_cache_gb=2.)
    print('echo_diagnosis_start '+json.dumps(dict(samples=len(ds),charts=len(ds.base.songs))),flush=True)
    result=evaluate_teacher_forced(model,ds,device=a.device,batch_size=a.batch_size,max_batches=a.max_batches,
        precision='fp32',loss_kwargs=dict(action_objective_policy=payload['args'].get('action_objective_policy','joint_event_recovery_v3'),
        audio_pointer_weight=payload['args'].get('audio_pointer_weight',.15),**action_group_loss_kwargs(model)))
    mass=result['echo_copy_mass_by_E']['4']['model_mass']
    result.update(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_epoch=payload['epoch'],optimizer_steps=payload['optimizer_steps'],
        manifest=str(Path(a.manifest).resolve()),elapsed_sec=time.perf_counter()-started,
        readonly=True,cache=ds.cache_stats(),decision=('insufficient_E4' if mass is None else
        'stop_report_sampling' if mass>=.6 else 'implement_capacity_prior' if mass<=.35 else 'implement_intermediate'))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
