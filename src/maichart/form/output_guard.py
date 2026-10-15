"""Hold an operating-system lock while a trainer writes one output directory."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def training_output_guard(directory):
    path=Path(directory).resolve()
    path.mkdir(parents=True,exist_ok=True)
    # Keep the lock file: deleting it can let a different process lock a new
    # inode while another trainer still holds the original file. OS locks
    # release on process exit, including cancellation or a crashed trainer.
    stream=(path/'.form_training.lock').open('a+b')
    acquired=False
    try:
        stream.seek(0,os.SEEK_END)
        if stream.tell()==0:
            stream.write(b'\0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f'已有训练任务正在写入此模型目录：{path}。请选择其他输出目录或先停止该任务。') from error
        acquired=True
        yield path
    finally:
        if acquired:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
