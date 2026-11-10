"""Background portable review export, preview and explicit import."""
from pathlib import Path
import re
import threading
import uuid

from maichart.training.dataset_catalog import _write_json

_LOCK=threading.RLock()


def post(endpoint,payload,*,runtime):
    operation=endpoint.rsplit('/',1)[-1]
    if operation not in {'export','preview','import'}:return 404,dict(error='未知片段包操作')
    root=runtime.WEBUI_ROOT/'review_bundles'
    if operation=='export':
        manifests=payload.get('manifests')
        candidates=payload.get('candidates')
        if not (isinstance(manifests,list) and manifests or isinstance(candidates,list) and candidates):
            raise ValueError('请勾选要导出的来源，或在完整谱面预览中导出本任务片段')
        if manifests and (not all(isinstance(p,str) and Path(p).is_file() for p in manifests)):
            raise ValueError('请选择有效的训练来源')
        request=dict(candidates=candidates) if candidates else dict(manifests=manifests)
    elif operation=='preview':
        path=Path(str(payload.get('path') or '')).resolve()
        if not path.is_file() or path.suffix.lower()!='.zip':raise ValueError('请选择已确认片段 ZIP 包')
        request=dict(path=str(path))
    else:
        identity=payload.get('preview_id')
        if not isinstance(identity,str) or not re.fullmatch('[a-f0-9]{32}',identity):raise ValueError('请先选择片段包并查看导入清单')
        source=root/identity/'result.json'
        import json
        if not source.is_file():raise ValueError('导入预览不存在，请重新选择片段包')
        request=json.loads(source.read_text(encoding='utf-8'))
        if request.get('kind')!='preview':raise ValueError('请选择已校验的导入清单')
    with _LOCK:
        if operation=='import':
            for task in runtime._all_tasks():
                if task.kind=='review_bundle_import' and task.status in ('queued','running'):
                    raise ValueError('已有片段包正在导入，请等待完成后再导入下一份')
        identity=uuid.uuid4().hex
        job=root/identity;request_file=job/'request.json';output=job/'result.json'
        _write_json(request_file,request)
        task=runtime._create_task('review_bundle_'+operation)
        task.result.update(review_bundle_id=identity)
        runtime._run_subprocess_task(task,['-m','maichart.training.review_bundle',operation,
            '--request',str(request_file),'--output',str(output),'--workspace-root',str(runtime.WEBUI_ROOT)],
            expected_epoch=None,result_paths=[output])
    return 200,dict(task_id=task.task_id,operation=operation,preview_id=identity if operation=='preview' else None)
