"""Persistent cache-discovery hints; selected artifacts still get full checks."""
from pathlib import Path
import json,sqlite3


class DiscoveryIndex:
    def __init__(self,path):
        self.connection=None
        try:
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            self.connection=sqlite3.connect(path,timeout=2,check_same_thread=False,isolation_level=None)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS entries (path TEXT PRIMARY KEY, signature TEXT, identity TEXT)')
        except (OSError,sqlite3.Error):self.connection=None

    def read(self,path,reader):
        path=Path(path);stat=path.stat()
        signature=json.dumps([stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns])
        try:
            if self.connection is not None:
                row=self.connection.execute('SELECT signature,identity FROM entries WHERE path=?',(str(path),)).fetchone()
                if row and row[0]==signature:return json.loads(row[1])
        except (sqlite3.Error,ValueError):pass
        identity=reader(path)
        try:
            if self.connection is not None:
                self.connection.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?)',
                    (str(path),signature,json.dumps(identity,ensure_ascii=False)))
        except (sqlite3.Error,TypeError,ValueError):pass
        return identity

    def close(self):
        if self.connection is not None:self.connection.close();self.connection=None
