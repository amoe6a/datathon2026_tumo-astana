"""Experimental loopback API entrypoint; preserves backend routes and Settings.

Start the separately supervised native model server first. Backend owner may
later add bonsai to Settings.from_env's engine allowlist and process lifecycle.
"""
import argparse
import os
from pathlib import Path
import uvicorn
from app.config import Settings,ROOT
from app.main import create_app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='models/bonsai/inference.json')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--runtime',default='runtime/bonsai-api')
    args=parser.parse_args()
    key=os.environ.get('FORTE_API_KEY','')
    if len(key)<16:parser.error('Set a private FORTE_API_KEY of at least 16 characters')
    settings=Settings(key,'bonsai',Path(args.config).resolve(),Path(args.runtime).resolve(),ROOT/'rules/catalog.json')
    uvicorn.run(create_app(settings),host='127.0.0.1',port=args.port,workers=1)


if __name__=='__main__':main()
