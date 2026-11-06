"""Keep diagnostic evidence immutable even when users reuse an output path."""
from pathlib import Path
import json
import os
import uuid


def write_failure_snapshot(output,payload):
    return _write_snapshot(output,payload,'failure')


def write_conflict_snapshot(output,payload):
    return _write_snapshot(output,payload,'conflicts')


def _write_snapshot(output,payload,kind):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    unique=output.with_name(output.stem+'.'+uuid.uuid4().hex+'.'+kind+'.json')
    latest=output.with_suffix('.'+kind+'.json')
    data={**payload,kind+'_snapshot_path':str(unique),kind+'_latest_path':str(latest)}
    text=json.dumps(data,ensure_ascii=False,indent=2)
    with unique.open('x',encoding='utf-8') as stream:stream.write(text)
    temporary=latest.with_name(latest.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(text,encoding='utf-8');os.replace(temporary,latest)
    except OSError as error:
        print('composer_v2_diagnostic_mirror_error '+json.dumps(dict(snapshot=str(unique),error=str(error))),flush=True)
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass  # The immutable snapshot is already safely written.
    return unique
