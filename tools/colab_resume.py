"""Resume an exported Composer bundle with its original learning contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def training_args(bundle, output, *, updates=0, data_workers=0):
    from maichart.composer.stream_train import parser as training_parser
    from maichart.portable_paths import material_path
    metadata = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    # Replay only arguments recognized by the current CLI, preserving every
    # saved learning/evaluation option instead of falling back to new defaults.
    parser = training_parser()
    args = parser.parse_args(['--manifest', 'pending', '--checkpoint-dir', str(output)])
    for key, value in metadata['args'].items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key in ('manifest', 'validation_manifest'):
        value = metadata['args'].get(key)
        setattr(args, key, str(material_path(value)) if value else None)
    args.checkpoint_dir = str(output)
    args.cache_dir = str(bundle / 'cache')
    args.resume = str(output / 'composer_last.pt' if (output / 'composer_last.pt').is_file()
                      else bundle / 'checkpoint/composer_last.pt')
    args.init_checkpoint = None
    args.data_workers = data_workers
    args.device = 'cuda'
    args.precision = 'bf16'
    args.stop_after_updates = updates
    args.rebatch_resume = False
    args.reset_learning_rate = False
    args.checkpoint_interval_steps = 100
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=0)
    parser.add_argument('--data-workers', type=int, choices=(0, 1), default=0)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--backup-dir', type=Path, help='Drive folder for synchronous checkpoint backups')
    options = parser.parse_args()
    bundle = options.bundle.resolve()
    mapping = json.loads((bundle / 'bundle.json').read_text(encoding='utf-8'))
    mapping = {letter + ':/': str(bundle / 'materials' / letter) for letter in mapping['drive_letters']}
    mapping_path = bundle / 'path_map.json'
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding='utf-8')
    os.environ['MAICHART_PATH_MAP'] = str(mapping_path)
    args = training_args(bundle, options.output.resolve(), updates=options.updates, data_workers=options.data_workers)
    if options.check_only:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('This BF16 resume requires an Ampere-or-newer CUDA GPU (e.g. A100/L4).')
    from maichart.composer.stream_train import _train
    from maichart.composer.training_lock import training_output_lock
    if options.backup_dir:
        from maichart.composer import stream_checkpoint
        backup = options.backup_dir.resolve()
        if backup == options.output.resolve():
            raise ValueError('Use a local output directory and a separate Drive backup directory')
        backup.mkdir(parents=True, exist_ok=True)
        original_save = stream_checkpoint.save

        def save_and_backup(path, *positional, **keywords):
            result = original_save(path, *positional, **keywords)
            target = backup / Path(path).name
            temporary = target.with_suffix(target.suffix + '.tmp')
            shutil.copyfile(path, temporary)
            temporary.replace(target)
            print('colab_checkpoint_backup ' + str(target), flush=True)
            return result

        stream_checkpoint.save = save_and_backup
    with training_output_lock(args.checkpoint_dir):
        _train(args)


if __name__ == '__main__':
    main()
