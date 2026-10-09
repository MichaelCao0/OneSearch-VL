"""Media preparation only. No semantic AV model is configured in Phase 1."""
from research.av_data.media_clock import clip_media, probe_media

CAPABILITIES = {
    'media_decode_ready': True,
    'raw_av_observer_ready': False,
    'asr_ready': False,
    'visual_semantic_observer_ready': False,
    'mode': 'media_receipt_only',
    'model': None,
    'model_revision': None,
    'semantic_probe_count': 0,
    'full_decode_work_accounting_ready': False,
}


def prepare_av_input(source, start_s, end_s, output_dir):
    derived = clip_media(source, start_s, end_s, output_dir)
    decoded = probe_media(derived['output_path'])
    streams = {s['index']: s for s in decoded['streams']}
    video = [f for f in decoded['frames'] if f['media_type'] == 'video']
    audio = [f for f in decoded['frames'] if f['media_type'] == 'audio']
    counts = {
        'decoded_audio_samples': sum(int(f.get('nb_samples', 0)) for f in audio),
        'returned_video_frames': len(video),
        'returned_video_pixels': sum(int(streams[f['stream_index']].get('width', 0)) * int(streams[f['stream_index']].get('height', 0)) for f in video),
        'output_bytes': derived['output_bytes'],
        'count_scope': 'prepared_clip_only',
        'full_source_decode_work': 'not_measured',
    }
    return derived, counts
