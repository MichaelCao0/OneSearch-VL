"""Public receipts describe observed bytes or publisher content, not gold."""
from copy import deepcopy
from .observer import CAPABILITIES, prepare_av_input
from research.av_search.retrieval import get_source


def provenance(index_dir, source_id, scope):
    doc = get_source(index_dir, source_id, scope)
    return {
        'status': 'ok', 'source_id': source_id,
        'origin_url': doc['origin_url'],
        'asset_sha256': doc['asset_sha256'],
        'text_sha256': doc['text_sha256'],
        'text_kind': doc['text_kind'],
        'public_metadata': doc['public_metadata'],
        'provenance_basis': 'frozen_public_publisher_content_and_asset_identity',
        'relationship_status': 'unknown',
    }


def compare_sources(index_dir, source_ids, scope):
    return {
        'status': 'ok',
        'sources': [provenance(index_dir, sid, scope) for sid in source_ids],
        'relationship_status': 'unknown',
        'comparison_basis': 'public_metadata_only',
    }


def inspect_av(source, start_s, end_s, output_dir):
    derived, counts = prepare_av_input(source, start_s, end_s, output_dir)
    public = {
        'status': 'media_ready_observer_unavailable',
        'source_id': source['source_id'], 'asset_sha256': source['asset_sha256'],
        'output_sha256': derived['output_sha256'],
        'original_interval': derived['original_interval'],
        'mapping': deepcopy(derived['mapping']),
        'transform': deepcopy(derived['transform']),
        'tool_version': derived['tool_version'],
        'capabilities': deepcopy(CAPABILITIES),
        'semantic_status': 'not_evaluated',
        'actual_cost': counts,
    }
    return public, derived


def read_text(index_dir, source_id, start_char, end_char, scope):
    """Return exact text from the pinned public snapshot; never a model summary."""
    doc = get_source(index_dir, source_id, scope)
    if (any(isinstance(x,bool) or not isinstance(x,int) for x in (start_char,end_char)) or
            not 0 <= start_char < end_char <= len(doc['text']) or end_char-start_char > 4000):
        raise ValueError('Text quote requires 1..4000 chars inside frozen content')
    return {'status':'ok','source_id':source_id,'origin_url':doc['origin_url'],
            'asset_sha256':doc['asset_sha256'],'text_sha256':doc['text_sha256'],
            'text_kind':doc['text_kind'],
            'quote':{'start_char':start_char,'end_char':end_char,'text':doc['text'][start_char:end_char]},
            'interpretation':'frozen_source_text_not_semantic_verification',
            'logical_cost':{'returned_chars':end_char-start_char}}
