"""Build a source-only handoff from Git-visible project files, never runtime data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FOLDERS={'app','ml','scripts','tests','web','configs','rules','docs','team','reports'}
SUFFIXES={'.py','.md','.txt','.json','.html','.css','.js','.cjs','.ps1','.sh','.svg','.webp','.jpg','.png'}
ROOT_FILES={'.gitignore','.dockerignore','.env.example','Dockerfile','LICENSE','AGENTS.md',
            'README.md','HACKATHON_PLAN.md','ML_PLAN.md','SOURCES.md','TEST_REPORT.md',
            'data.example.json','run.ps1'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='team-kit/forte-review-team-kit.zip')
    args=parser.parse_args()
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    selected=[]
    for name in sorted(set(names)):
        p=Path(name)
        if not name or not (ROOT/p).is_file():continue
        if len(p.parts)>1 and p.parts[0] not in FOLDERS:continue
        if len(p.parts)==1 and name not in ROOT_FILES and not (name.startswith('requirements') and p.suffix=='.txt'):continue
        if p.suffix.lower() not in SUFFIXES and name not in ROOT_FILES:continue
        if not (ROOT/p).resolve().is_relative_to(ROOT):raise ValueError('Source file resolves outside project')
        if any(part in {'__pycache__','runtime','data','models','.venv'} for part in p.parts):continue
        if any(token in p.name.lower() for token in ['access-key','private-','service.env']):continue
        selected.append(p)
    output=(ROOT/args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    manifest={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'scope':'Source and public assets only. Generate synthetic fixtures locally. Download weights separately.',
              'files':{p.as_posix():hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in selected}}
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in selected:archive.write(ROOT/p,p.as_posix())
        archive.writestr('KIT_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(selected)+1
    print(json.dumps({'archive':str(output),'files':len(selected),'bytes':output.stat().st_size,
                      'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
