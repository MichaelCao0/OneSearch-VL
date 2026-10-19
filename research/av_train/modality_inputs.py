"""Physically remove audio from every model-visible video in muted evaluation."""
from pathlib import Path
from copy import deepcopy
import json, subprocess, hashlib

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def streams(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=30)
    return [s['codec_type'] for s in json.loads(p.stdout)['streams']]

def video_packet_sha(path):
    return subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-c:v', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], check=True, capture_output=True, text=True, timeout=120).stdout.strip()

class MutedMedia:

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    def video(self, source):
        original = Path(source).resolve()
        if not original.is_file():
            raise ValueError('Ablation requires frozen local media')
        stat = original.stat()
        cache_key = (str(original), stat.st_size, stat.st_mtime_ns)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if 'audio' not in streams(original):
            self.cache[cache_key] = str(original)
            return str(original)
        h = sha(original)
        suffix = '.mp4' if original.suffix.lower() in {'.mp4', '.mov', '.m4v'} else '.mkv'
        dest = self.folder / (h + suffix)
        receipt = self.folder / (h + '.json')
        if dest.exists():
            r = json.loads(receipt.read_text())
            assert r['source_sha256'] == h and r['muted_sha256'] == sha(dest)
        else:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original), '-map', '0:v:0', '-c:v', 'copy', '-an', '-map_metadata', '-1'] + (['-movflags', '+faststart'] if suffix == '.mp4' else []) + [str(dest)], check=True, capture_output=True, timeout=120)
            before = video_packet_sha(original)
            after = video_packet_sha(dest)
            assert before == after
            receipt.write_text(json.dumps({'source_sha256': h, 'muted_sha256': sha(dest), 'video_packet_sha256': before, 'encoded_video_packets_identical': True, 'operation': 'copy_first_video_stream_drop_all_audio', 'no_audio_stream': True}, indent=2) + '\n')
        assert streams(dest) == ['video']
        self.cache[cache_key] = str(dest)
        return str(dest)

    def contents(self, contents):
        result = deepcopy(contents)
        count = 0
        for m in result:
            for part in m.get('parts', []):
                if 'video_url' in part or 'video' in part:
                    key = 'video_url' if 'video_url' in part else 'video'
                    v = part[key]
                    meta = v if isinstance(v, dict) else {'url': v}
                    urlkey = 'url' if 'url' in meta else 'video'
                    meta[urlkey] = self.video(meta[urlkey])
                    meta['has_audio'] = False
                    part[key] = meta
                    count += 1
                if 'audio' in part or 'audio_url' in part or part.get('inline_data', {}).get('mime_type', '').startswith('audio/'):
                    raise ValueError('Unexpected explicit audio would contaminate muted ablation')
        if count == 0:
            raise ValueError('Native AV history must retain at least one video')
        return result
