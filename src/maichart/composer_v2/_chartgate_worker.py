"""JSON-lines subprocess adapter for the upstream CPU runtime."""
from contextlib import redirect_stdout, suppress
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    root, dll, dependencies = map(Path, sys.argv[1:])
    sys.path.insert(0, str(root/'wwwroot/python'))
    sys.path.append(str(dependencies))
    with redirect_stdout(sys.stderr):
        from maimai_style import browser_runtime
        model = browser_runtime.initialize_model((root/'wwwroot/model/model.json').read_text(encoding='utf-8'))
    with subprocess.Popen(['dotnet', str(dll)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          encoding='utf-8', creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as reader:
        try:
            print(json.dumps({'ready':True}),flush=True)
            for line in sys.stdin:
                reader.stdin.write(line)
                reader.stdin.flush()
                response = reader.stdout.readline()
                if not response:
                    raise RuntimeError('Chart;Gate 谱面解析进程已退出。')
                try:
                    record = json.loads(response)
                    if 'error' in record:
                        raise ValueError(record['error'])
                    if not any(group['notes'] for group in record['timings']):
                        raise ValueError('谱面没有音符')
                    with redirect_stdout(sys.stderr):
                        features, _ = browser_runtime.structural_feature_payload(
                            record, feature_schema_version=model['feature_schema_version'])
                        level = float(browser_runtime._predict_difficulty(features)['point'])
                    if not math.isfinite(level):
                        raise ValueError('定数无效')
                    result = {'level':level}
                except (ValueError,ArithmeticError,TypeError,KeyError,IndexError,RuntimeError) as error:
                    result = {'error':str(error)}
                print(json.dumps(result,ensure_ascii=True,allow_nan=False),flush=True)
        finally:
            with suppress(BrokenPipeError):
                reader.stdin.close()


if __name__ == '__main__':
    main()
