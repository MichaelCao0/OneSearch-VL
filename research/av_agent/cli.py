"""Replay explicit JSONL tool actions against a trusted frozen source registry."""
import argparse
import json
from pathlib import Path
from .adapter import EpisodeTools, strict_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--actions', required=True)
    parser.add_argument('--max-calls', type=int, default=20)
    parser.add_argument('--max-av-seconds', type=float, default=60)
    args=parser.parse_args()
    records=[strict_json(line, max_chars=16*1024*1024) for line in Path(args.registry).read_text().splitlines() if line.strip()]
    registry={record['source_id']:record for record in records}
    if len(registry)!=len(records):
        raise ValueError('Duplicate source registry identities')
    tools=EpisodeTools(args.index_dir,registry,strict_json(Path(args.scope).read_text()),args.output_dir,args.max_calls,args.max_av_seconds)
    for line in Path(args.actions).read_text().splitlines():
        if line.strip():
            print(json.dumps(tools.dispatch(line),ensure_ascii=False,allow_nan=False))
    (Path(args.output_dir)/'costs.json').write_text(json.dumps(tools.costs,indent=2))

if __name__=='__main__':
    main()
