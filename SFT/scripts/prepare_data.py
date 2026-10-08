"""Prepare ShareGPT JSON with one dataset per initial visual input type."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

IMAGE_SOURCES = (
    'fvqa_agent_sft_v3', 'palace_agent_sft_v3', 'webqa_agent_sft_v3',
    'livevqa_agent_sft_v3', 'wikiart_agent_sft_v3', 'wiki_en_agent_sft_v3',
    'wiki_zh_agent_sft_v3',
)
LAYOUT = {
    'video_agent_sft': 'video/video_llama_factory.json',
    'multi_image_agent_sft': 'multi_image/multi_image_llama_factory.json',
    'image_agent_sft': 'image/image_llama_factory.json',
}


def json_rows(path):
    decoder = json.JSONDecoder()
    with open(path) as f:
        buf = ''; pos = 0; eof = False; opened = False
        while True:
            while pos < len(buf) and buf[pos] in ' \r\n\t,':
                pos += 1
            if not opened and pos < len(buf):
                if buf[pos] != '[':
                    raise ValueError('Expected JSON array')
                opened = True; pos += 1; continue
            if opened and pos < len(buf) and buf[pos] == ']':
                return
            try:
                row, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError('Truncated JSON array')
                chunk = f.read(4 * 1024 * 1024)
                buf = buf[pos:] + chunk; pos = 0; eof = not chunk
                continue
            yield row; pos = end


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_array(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('x') as f:
        f.write('[\n')
        for row in rows:
            if n:
                f.write(',\n')
            f.write(json.dumps(row, ensure_ascii=False))
            n += 1
        f.write('\n]\n')
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--from-jsonl', action='store_true', help='Convert the audited JSONL source bundle')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--media-root', type=Path, help='Directory containing media/')
    group.add_argument('--source-map', type=Path, help='Local path mapping; never included in portable data')
    args = ap.parse_args()
    info = json.loads((args.input_dir / 'dataset_info.json').read_text())
    sources = {r['path']: r['source'] for r in read_jsonl(args.source_map)} if args.source_map else None
    @lru_cache(maxsize=None)
    def resolve(path):
        if sources is not None:
            return sources[path]
        if args.media_root:
            p = Path(path)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'media':
                raise ValueError(f'Invalid media path: {path}')
            result = args.media_root.resolve() / p
            if not result.is_file():
                raise FileNotFoundError(result)
            return str(result)
        return path
    output_info = {}
    for name, rel in LAYOUT.items():
        names = [name] if name in info else list(IMAGE_SOURCES) if name == 'image_agent_sft' else [name]
        registration = dict(info[names[0]])
        def converted():
            for source_name in names:
                entry = info[source_name]
                if any(entry[k] != registration[k] for k in ('formatting', 'columns', 'tags')):
                    raise ValueError('Cannot merge incompatible ShareGPT registrations')
                reader = read_jsonl if args.from_jsonl else json_rows
                source_path = args.input_dir / entry['file_name']
                if args.from_jsonl and source_path.suffix == '.json':
                    source_path = source_path.with_suffix('.jsonl')
                for row in reader(source_path):
                    for field in ('images','videos'):
                        if row.get(field) is not None:
                            row[field] = [resolve(v) for v in row[field]]
                    yield row
        count = write_array(args.output_dir / rel, converted())
        registration['file_name'] = rel
        output_info[name] = registration
        print(name, count, flush=True)
    (args.output_dir / 'dataset_info.json').write_text(json.dumps(output_info, indent=2)+'\n')

if __name__ == '__main__':
    main()
