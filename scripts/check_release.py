"""Verify public file hashes, Python syntax and static local dependencies."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    metadata=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
    for entry in metadata['files']:
        path=ROOT/entry['path']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256'],path
    code=ROOT/'code';count=0
    for path in ROOT.rglob('*.py'):
        if '.git' in path.parts:continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));count+=1
        for node in ast.walk(tree):
            modules=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                if module in metadata['local_modules']:assert (code/(module+'.py')).is_file(),module
    print(f'PASS: {len(metadata["files"])} file hashes; {count} Python syntax checks.')


if __name__=='__main__':main()
