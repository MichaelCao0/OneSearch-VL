"""Export and audit native Omni trajectories for the original LLaMA-Factory SFT.

No custom optimizer, model construction, GPU execution or synthetic teaching
trajectories. Development episodes are permitted only in an isolated audit.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import MethodType, SimpleNamespace
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "native-omni-paired-sft.v1"

from .generation_receipts import generation_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def local_media(source):
    if not isinstance(source, str) or source.startswith("<"):
        raise ValueError("stripped or unavailable media cannot enter native SFT")
    parsed = urlparse(source)
    if parsed.scheme not in {"", "file"}:
        raise ValueError("native SFT requires a frozen local media file")
    path = Path(unquote(parsed.path) if parsed.scheme else source).resolve()
    if not path.is_file():
        raise ValueError("native SFT media file is unavailable")
    return str(path)


def validate_trajectory(trace):
    """Reconstruct the last accumulated snapshot; verify every actual action."""
    if trace.get("input_mode") != "native_audio_video" or trace.get("modality") != "audio_video":
        raise ValueError("genuine native audio/video pipeline trajectory required")
    if trace.get("status") != "ok" or not trace.get("turns"):
        raise ValueError("only completed original pipeline trajectories can be exported")
    if trace.get("protocol_repair") or any(turn.get("protocol_repair") for turn in trace["turns"]):
        raise ValueError("protocol repair trajectories are excluded from initial native SFT")
    previous = []
    teacher_identity = None
    for index, turn in enumerate(trace["turns"]):
        snapshot = turn.get("chat_completions")
        text = turn.get("response_text")
        if turn.get("error") or not isinstance(snapshot, list) or not text:
            raise ValueError("incomplete or failed generation cannot be exported")
        if not snapshot or snapshot[0].get("role") != "system" or snapshot[-1] != {"role": "assistant", "content": text}:
            raise ValueError("snapshot/output mismatch")
        if previous and (snapshot[:len(previous)] != previous or len(snapshot) != len(previous)+2):
            raise ValueError("accumulated history lost/changed or observations missing")
        candidates = turn.get("response", {}).get("candidates", [])
        if not candidates or candidates[0].get("finishReason") == "ERROR":
            raise ValueError("actual successful model response required")
        rendered = "".join(p.get("text", "") for c in candidates for p in c.get("content", {}).get("parts", []))
        if rendered != text:
            raise ValueError("actual response text was replaced")
        identity = generation_identity(turn["response"], snapshot)
        if teacher_identity is not None and identity != teacher_identity:
            raise ValueError("teacher model identity changed within a trajectory")
        teacher_identity = identity
        calls = list(re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S))
        if index < len(trace["turns"])-1:
            if (len(calls) != 1 or text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1
                    or "<response>" in text or "</response>" in text):
                raise ValueError("nonfinal assistant must contain exactly one executed action and no final answer")
            try:
                action = json.loads(calls[0].group(1))
            except (ValueError, TypeError) as exc:
                raise ValueError("executed action must contain valid tool JSON") from exc
            if (not isinstance(action, dict) or not isinstance(action.get("name"), str)
                    or not action["name"].strip() or not isinstance(action.get("arguments"), dict)):
                raise ValueError("executed action requires a tool name and argument object")
            output = turn.get("tool_output")
            if not isinstance(output, str) or not output.strip():
                raise ValueError("executed action must have its recorded tool output")
            following = trace["turns"][index+1].get("chat_completions", [])
            if len(following) != len(snapshot)+2:
                raise ValueError("executed action is missing its next observation")
            observation = following[len(snapshot)]
            content = observation.get("content")
            observation_text = (content if isinstance(content, str) else
                                "".join(p.get("text", "") for p in content if p.get("type") == "text")
                                if isinstance(content, list) else None)
            if observation.get("role") != "user" or observation_text != f"<tool_response>\n{output}\n</tool_response>":
                raise ValueError("recorded tool output does not match the next model observation")
        elif "<tool_call>" in text or "</tool_call>" in text or turn.get("tool_output"):
            raise ValueError("final response cannot supervise an unexecuted tool action")
        previous = deepcopy(snapshot)
    final = trace["turns"][-1]["response_text"]
    responses = list(re.finditer(r"<response>(.+?)</response>", final, re.S))
    if len(responses) != 1 or not responses[0].group(1).strip():
        raise ValueError("completed native final response required")
    final_answer = responses[0].group()
    if trace.get("final_response_text") != final_answer:
        raise ValueError("final answer does not match real generation")
    body = previous[1:]
    if len(body) % 2 or any(m.get("role") != ("user" if i % 2 == 0 else "assistant") for i,m in enumerate(body)):
        raise ValueError("original alternating user/assistant snapshots required")
    return previous


def audio_sidecar(video, media_dir, cache):
    """Extract real recorded audio only; no silence fabrication or resynthesis."""
    video = local_media(video)
    if video in cache:
        return cache[video]
    receipt = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                              "-of", "json", video], check=True, capture_output=True, text=True, timeout=30)
    if not json.loads(receipt.stdout).get("streams"):
        cache[video] = (None, None)
        return cache[video]
    source_hash = file_sha(video)
    out = Path(media_dir)/(source_hash + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", video,
                        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
                       check=True, capture_output=True, timeout=180)
    receipt = {"source_sha256": source_hash, "audio_sha256": file_sha(out),
               "audio_path": str(out.resolve()), "sample_rate": 16000,
               "operation": "decode_first_recorded_audio_stream_mono_pcm_s16le", "fabricated_silence": False}
    cache[video] = (str(out.resolve()), receipt)
    return cache[video]


def snapshot_row(snapshot, audio_resolver):
    """Turn OAI media blocks into LF placeholders and original media arrays."""
    row = {"system": snapshot[0]["content"], "tools": "", "conversations": [], "images": [], "videos": [], "audios": []}
    audio_presence, receipts = [], []
    for message in snapshot[1:]:
        content = message["content"]
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            blocks = []
            for block in content:
                kind = block.get("type")
                if kind == "text":
                    blocks.append(block["text"])
                elif kind == "video_url":
                    path = local_media(block["video_url"]["url"])
                    if block.get("native_av") is not True:
                        raise ValueError("raw native AV receipt required for every video")
                    row["videos"].append(path)
                    audio, receipt = audio_resolver(path)
                    audio_presence.append(audio is not None)
                    blocks.append("<video>" + ("<audio>" if audio is not None else ""))
                    if audio is not None:
                        row["audios"].append(audio)
                        receipts.append(receipt)
                elif kind == "image_url":
                    row["images"].append(local_media(block["image_url"]["url"]))
                    blocks.append("<image>")
                else:
                    raise ValueError("unsupported or stripped snapshot media cannot be dropped")
            text = "\n".join(blocks)
        else:
            raise ValueError("unsupported conversation content")
        if any(marker in text for marker in ("<inline_video_stripped>", "<base64_image_data:", "<base64_video_data:")):
            raise ValueError("snapshot media payload is stripped")
        row["conversations"].append({"from": "human" if message["role"] == "user" else "gpt", "value": text})
    if not row["videos"]:
        raise ValueError("initial native AV is required")
    # Audio stream metadata is separate from model text and training columns.
    return row, audio_presence, receipts


def answer_only_row(snapshot, audio_resolver):
    final = snapshot[-1]
    match = re.search(r"<response>.+?</response>", final["content"], re.S)
    if match is None:
        raise ValueError("actual generated final answer required")
    paired = [snapshot[0], snapshot[1], {"role":"assistant", "content":match.group()}]
    return snapshot_row(paired, audio_resolver)


def dataset_registration(filename):
    return {"file_name":filename, "formatting":"sharegpt",
            "columns":{"messages":"conversations", "system":"system", "tools":"tools", "images":"images", "videos":"videos", "audios":"audios"},
            "tags":{"role_tag":"from", "content_tag":"value", "user_tag":"human", "assistant_tag":"gpt", "observation_tag":"observation", "function_tag":"function_call", "system_tag":"system"}}


def export_episodes(episodes_dir, out_dir, *, verdicts=None, audit_only=False, case_ids=None):
    episodes = Path(episodes_dir).resolve()
    inputs = json.loads((episodes/"inputs.private.json").read_text())
    batch = Path(inputs["batch_dir"])
    if not batch.is_absolute():
        batch = ROOT/batch
    scope = json.loads((batch/"scope.json").read_text())
    tasks = {t["task_id"]:t for t in (json.loads(line) for line in (batch/"tasks-public.jsonl").read_text().splitlines() if line)}
    bindings = {c["case_id"]:c for c in inputs["cases"]}
    clue_bindings = {c["clue_id"]:c["source"] for c in json.loads((batch/"clue-bindings.private.json").read_text())}
    if not audit_only and (scope.get("split") != "train" or any(t.get("split") != "train" for t in tasks.values())):
        raise ValueError("development/test batches are audit-only and cannot become training data")
    if not audit_only and not verdicts:
        raise ValueError("successful retention verdicts required; pipeline status ok alone is insufficient")
    if case_ids and set(case_ids)-set(bindings):
        raise ValueError("requested case absent from episode input bindings")
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    media_dir = out/"audio"; media_dir.mkdir()
    cache = {}; full = []; paired = []; lineage = []; rejected = []; availability = []
    for path in sorted(episodes.rglob("*_trajectory.json")):
        trace = json.loads(path.read_text()); cid = trace.get("case_id")
        if case_ids and cid not in case_ids:
            continue
        try:
            if cid not in bindings:
                raise ValueError("trajectory absent from authorized batch/task bindings")
            binding = bindings[cid]
            task_id = binding.get("task_id", cid)
            if task_id not in tasks:
                raise ValueError("candidate task absent from authorized batch/task bindings")
            task = tasks[task_id]
            initial_clues = task.get("initial_sources", [])
            if len(initial_clues) != 1 or initial_clues[0] not in clue_bindings:
                raise ValueError("authoritative anonymous clue binding required")
            source = clue_bindings[initial_clues[0]]
            if (Path(source["asset_path"]).resolve() != Path(binding["source_path"]).resolve()
                    or source["asset_sha256"] != binding["source_sha256"]):
                raise ValueError("initial media does not belong to the batch's split-bound clue")
            if (trace.get("question") != binding["question"] or trace["question"] != task["question"]
                    or Path(trace["input_media"]["path"]).resolve() != Path(binding["source_path"]).resolve()
                    or file_sha(binding["source_path"]) != binding["source_sha256"]):
                raise ValueError("initial task/media binding mismatch")
            if not audit_only:
                verdict = verdicts.get(cid, {})
                if (verdict.get("success") is not True or verdict.get("split") != "train"
                        or not verdict.get("reviewer_id") or verdict.get("trajectory_sha256") != file_sha(path)
                        or verdict.get("quality_tier") not in {"silver", "human_verified"}):
                    raise ValueError("successful split/hash-bound retention verdict required")
            snapshot = validate_trajectory(trace)
            for message in snapshot[1:]:
                for block in message["content"] if isinstance(message["content"],list) else []:
                    if block.get("type")=="video_url":
                        media_path = Path(local_media(block["video_url"]["url"]))
                        initial_path = Path(binding["source_path"]).resolve()
                        if media_path != initial_path and not media_path.is_relative_to(path.parent.resolve()/"intermediate"):
                            raise ValueError("video outside the bound initial clue or this episode's derived clips")
            initial = snapshot[1].get("content")
            if not isinstance(initial,list) or not any(b.get("type")=="video_url" and Path(b["video_url"]["url"]).resolve()==Path(binding["source_path"]).resolve() for b in initial):
                raise ValueError("initial native media missing from actual snapshot")
            resolver = lambda media: audio_sidecar(media, media_dir, cache)
            row, sounding, receipts = snapshot_row(snapshot, resolver)
            answer, _, _ = answer_only_row(snapshot, resolver)
            full.append(row); paired.append(answer); availability.extend(sounding)
            lineage.append({"row_index":len(full)-1,"case_id":cid,"task_id":task_id,"split":task["split"],
                            "teacher_identity":generation_identity(trace["turns"][0]["response"],trace["turns"][0]["chat_completions"]),
                            "trajectory_path":str(path),"trajectory_sha256":file_sha(path),
                            "row_sha256":digest(row),"answer_row_sha256":digest(answer),
                            "assistant_turns":len(trace["turns"]),"retention_verdict_sha256":digest(verdicts[cid]) if not audit_only else None,
                            "media_audio_presence":sounding,"audio_receipts":receipts})
        except (ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            rejected.append({"case_id":cid,"trajectory_sha256":file_sha(path),"reason":str(exc)})
    use_joint = bool(availability) and all(availability)
    write_json(out/"trajectory.json",full); write_json(out/"answer_only.json",paired)
    write_json(out/"lineage.private.json",lineage); write_json(out/"rejections.private.json",rejected)
    if not audit_only:
        write_json(out/"dataset_info.json",{"native_omni_trajectory":dataset_registration("trajectory.json"),
                                         "native_omni_answer_only":dataset_registration("answer_only.json")})
    summary = {"schema_version":SCHEMA,"audit_only":audit_only,"source_split":scope["split"],
               "batch_dir":str(batch),"episodes_dir":str(episodes),"retained_count":len(full),"rejected_count":len(rejected),
               "distinct_task_count":len({v["task_id"] for v in lineage}),
               "use_audio_in_video":use_joint,"media_mode":"joint_av" if use_joint else "native_separate_streams",
               "retained_pool_sha256":digest([v["case_id"] for v in lineage]),"trajectory_rows_sha256":digest(full),
               "answer_rows_sha256":digest(paired),"all_assistant_actions_supervised":True,
               "answers_are_actual_generated_responses":True,"train_on_prompt":False,"mask_history":False,
               "training_ready":False,"cpu_label_media_audit":"required","model_loaded":False,"optimizer_updates":0,
               "retention_success_verdicts_checked":not audit_only,"benchmark_gold_verified":False}
    write_json(out/"summary.json",summary)
    return summary


def native_components(snapshot, cutoff_len, use_audio_in_video):
    source = str(ROOT/"SFT/src")
    if source not in sys.path:
        sys.path.insert(0,source)
    from llamafactory.hparams import ModelArguments, DataArguments
    from llamafactory.model.loader import load_tokenizer
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.converter import SharegptDatasetConverter
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    model_args = ModelArguments(model_name_or_path=str(snapshot), trust_remote_code=False,
                               use_audio_in_video=use_audio_in_video, video_fps=2.,video_maxlen=128,
                               video_min_pixels=3136,video_max_pixels=100352,
                               image_min_pixels=3136,image_max_pixels=262144,audio_sampling_rate=16000)
    loaded = load_tokenizer(model_args)
    if loaded["processor"] is None or loaded["processor"].__class__.__name__ != "Qwen2_5OmniProcessor":
        raise ValueError("original loader did not load native Omni processor")
    data = DataArguments(template="qwen2_omni",cutoff_len=cutoff_len,train_on_prompt=False,mask_history=False,packing=False)
    attr = DatasetAttr("file","native-audit"); attr.join(dataset_registration("audit.json"))
    template = get_template_and_fix_tokenizer(loaded["tokenizer"],data)
    return (SharegptDatasetConverter(attr,data),template,
            SupervisedDatasetProcessor(template,loaded["tokenizer"],loaded["processor"],data),loaded)


def audit_dataset(dataset_dir, model_manifest, out_dir, *, cutoff_len=32768):
    """Run original converter/plugin/supervised processor/collator on CPU."""
    dataset_dir=Path(dataset_dir); summary=json.loads((dataset_dir/"summary.json").read_text())
    if summary.get("schema_version") != SCHEMA or summary.get("retained_count",0)<1:
        raise ValueError("nonempty native export required for audit")
    manifest=json.loads(Path(model_manifest).read_text())
    if manifest.get("repository_id")!="Qwen/Qwen2.5-Omni-7B" or manifest.get("status")!="verified":
        raise ValueError("verified local Omni manifest required")
    converter,template,processor,loaded=native_components(manifest["snapshot_path"],cutoff_len,summary["use_audio_in_video"])
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
    from transformers import Qwen2_5OmniConfig,Qwen2_5OmniThinkerForConditionalGeneration as Thinker
    config=Qwen2_5OmniConfig.from_pretrained(manifest["snapshot_path"],local_files_only=True).thinker_config
    rope=SimpleNamespace(config=config,spatial_merge_size=config.vision_config.spatial_merge_size)
    rope.get_llm_pos_ids_for_vision=MethodType(Thinker.get_llm_pos_ids_for_vision,rope)
    rope.get_chunked_index=MethodType(Thinker.get_chunked_index,rope)
    rope.get_rope_index=MethodType(Thinker.get_rope_index,rope)
    collator=MultiModalDataCollatorForSeq2Seq(tokenizer=loaded["tokenizer"],model=rope,template=template,
                                            processor=loaded["processor"],padding=True,label_pad_token_id=IGNORE_INDEX)
    reports=[]; coverage_risks=[]; coverage_receipts={}
    for dataset_name,filename in (("trajectory","trajectory.json"),("answer_only","answer_only.json")):
        rows=json.loads((dataset_dir/filename).read_text())
        expected_hash=summary["trajectory_rows_sha256" if dataset_name=="trajectory" else "answer_rows_sha256"]
        if digest(rows)!=expected_hash:
            raise ValueError("export rows changed before audit")
        for index,row in enumerate(rows):
            aligned=converter(row)
            for video in aligned["_videos"] or []:
                if video in coverage_receipts:
                    continue
                native=loaded["processor"]
                sampled=template.mm_plugin._regularize_videos([video],
                    image_min_pixels=native.video_min_pixels,image_max_pixels=native.video_max_pixels,
                    image_factor=native.video_processor.patch_size*native.video_processor.merge_size,
                    video_fps=native.video_fps,video_maxlen=native.video_maxlen)
                receipt=sampled["sampling_receipts"][0]
                verified=(receipt["selected_frame_indices"][0]==0
                          and receipt["selected_frame_indices"][-1]==receipt["decoded_frame_count"]-1
                          and 2<=receipt["selected_frame_count"]<=native.video_maxlen
                          and all(native.video_min_pixels<=w*h<=native.video_max_pixels
                                  for w,h in receipt["selected_frame_sizes_wh"])
                          and receipt["processor_do_resize"] is False)
                coverage_receipts[video]={**receipt,"video_path":video,"video_sha256":file_sha(video),
                                          "coverage_verified":verified}
                if not verified:
                    coverage_risks.append({"video_sha256":file_sha(video),"risk":"full_clip_or_pixel_budget_verification_failed"})
                del sampled
            if len(aligned["_prompt"]+aligned["_response"])!=len(row["conversations"]):
                raise ValueError("original converter discarded conversation turns")
            media=[aligned[k] or [] for k in ("_images","_videos","_audios")]
            expanded=template.mm_plugin.process_messages(aligned["_prompt"]+aligned["_response"],*media,loaded["processor"])
            pairs=template.encode_multiturn(loaded["tokenizer"],expanded,aligned["_system"],aligned["_tools"])
            expected=sum(len(source)+len(target) for source,target in pairs)+(1 if template.efficient_eos else 0)
            if expected>cutoff_len:
                raise ValueError("native sequence exceeds cutoff; never silently truncate actions/media")
            ids,labels=processor._encode_data_example(aligned["_prompt"],aligned["_response"],aligned["_system"],
                                                      aligned["_tools"],*media)
            if len(ids)!=expected:
                raise ValueError("supervised processor truncated or changed sequence")
            offset=0; turn_reports=[]
            for turn,(source,target) in enumerate(pairs):
                if labels[offset:offset+len(source)] != [IGNORE_INDEX]*len(source):
                    raise ValueError("user/system/observation media tokens are training targets")
                begin=offset+len(source); end=begin+len(target)
                if not target or labels[begin:end]!=target or ids[begin:end]!=target:
                    raise ValueError("assistant action/answer masked or lost")
                target_text=loaded["tokenizer"].decode(target,skip_special_tokens=False)
                raw=row["conversations"][turn*2+1]["value"]
                if raw not in target_text:
                    raise ValueError("actual assistant text not preserved in target tokens")
                turn_reports.append({"assistant_turn":turn,"source_masked_tokens":len(source),
                                     "target_tokens":len(target),"actual_text_sha256":hashlib.sha256(raw.encode()).hexdigest()})
                offset=end
            feature={"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels,
                     "images":media[0],"videos":media[1],"audios":media[2]}
            batch=collator([feature])
            if media[1] and "pixel_values_videos" not in batch:
                raise ValueError("collator dropped native video")
            if media[2] and ("input_features" not in batch or batch["input_features"].shape[0]!=len(media[2])):
                raise ValueError("collator dropped real audio")
            grid_values=batch["video_grid_thw"].tolist() if media[1] else []
            processed_pixels=[]
            if len(grid_values)!=len(media[1]):
                raise ValueError("original collator video grid count does not match native media")
            for video,(grid_t,grid_h,grid_w) in zip(media[1],grid_values):
                receipt=coverage_receipts[video]
                pixels=grid_h*grid_w*loaded["processor"].video_processor.patch_size**2
                if (grid_t*loaded["processor"].video_processor.temporal_patch_size!=receipt["selected_frame_count"]
                        or pixels!=receipt["selected_frame_sizes_wh"][0][0]*receipt["selected_frame_sizes_wh"][0][1]
                        or not receipt["configured_min_pixels"]<=pixels<=receipt["configured_max_pixels"]):
                    raise ValueError("original collator changed configured video frame count/pixel budget")
                processed_pixels.append(pixels)
            supervised=[token for token in batch["labels"][0].tolist() if token!=IGNORE_INDEX]
            if supervised != [token for token in labels if token!=IGNORE_INDEX]:
                raise ValueError("collator changed supervision labels")
            tensors={k:{"shape":list(v.shape),"dtype":str(v.dtype),"device":str(v.device)}
                     for k,v in batch.items() if hasattr(v,"shape") and hasattr(v,"device")}
            reports.append({"dataset":dataset_name,"row_index":index,"total_tokens":len(ids),
                            "supervised_tokens":len(supervised),"assistant_turns":len(pairs),"turns":turn_reports,
                            "video_count":len(media[1]),"audio_count":len(media[2]),"tensors":tensors,
                            "video_grid_thw_values":grid_values,"processed_frame_pixels":processed_pixels,
                            "video_second_per_grid_values":batch["video_second_per_grid"].tolist() if media[1] else [],
                            "sampled_video_sha256":[coverage_receipts[video]["video_sha256"] for video in media[1]]})
    files=["data/converter.py","data/template.py","data/mm_plugin.py","data/processor/supervised.py","data/collator.py","model/loader.py"]
    report={"schema_version":"native-omni-label-media-audit.v2","status":"passed","audit_only":summary["audit_only"],
            "dataset_summary_sha256":file_sha(dataset_dir/"summary.json"),"template":"qwen2_omni",
            "processor":"Qwen2_5OmniProcessor","use_audio_in_video":summary["use_audio_in_video"],
            "train_on_prompt":False,"mask_history":False,"all_assistant_turns_supervised":True,
            "user_observation_media_masked":True,"cutoff_len":cutoff_len,"rows":reports,
            "original_sampler_native_video_coverage_verified":bool(coverage_receipts) and not coverage_risks,
            "video_sampling_coverage_risks":coverage_risks,
            "video_sampling_coverage_receipts":list(coverage_receipts.values()),
            "processor_second_resize_disabled":True,"runtime_prefix_equivalence_claimed":False,
            "native_source_sha256":{f:file_sha(ROOT/"SFT/src/llamafactory"/f) for f in files},
            "packages":{p:importlib.metadata.version(p) for p in ("torch","transformers","datasets","peft","trl","qwen-omni-utils","librosa","decord")},
            "model_loaded":False,"optimizer_updates":0,"training_ready":False}
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False);write_json(out/"label-media-audit.private.json",report)
    return report


def materialize_configs(dataset_dir, audit_dir, model_manifest, out_dir):
    """Create launchable original-trainer configs only for audited retained train."""
    import yaml
    data=Path(dataset_dir).resolve();summary=json.loads((data/"summary.json").read_text())
    audit=json.loads((Path(audit_dir)/"label-media-audit.private.json").read_text())
    if (summary.get("audit_only") is not False or summary.get("source_split")!="train" or summary.get("retained_count",0)<1
            or audit.get("status")!="passed" or audit.get("audit_only") is not False
            or audit.get("dataset_summary_sha256")!=file_sha(data/"summary.json")):
        raise ValueError("audited nonempty successfully retained train pool required; dev audit cannot train")
    if audit.get("original_sampler_native_video_coverage_verified") is not True:
        raise ValueError("original trainer video sampler coverage requires resolution before training configs")
    receipts=audit.get("video_sampling_coverage_receipts",[])
    if not receipts or any(item.get("coverage_verified") is not True or item.get("processor_do_resize") is not False
                           for item in receipts):
        raise ValueError("verified source-bound video sampling receipts required")
    for source,expected in audit.get("native_source_sha256",{}).items():
        if file_sha(ROOT/"SFT/src/llamafactory"/source)!=expected:
            raise ValueError("original trainer source changed after native audit")
    if not audit.get("native_source_sha256",{}):
        raise ValueError("original trainer source hashes required")
    for receipt in receipts:
        if file_sha(receipt["video_path"])!=receipt["video_sha256"]:
            raise ValueError("audited video changed after sampling coverage verification")
    for name,filename in (("trajectory","trajectory.json"),("answer_only","answer_only.json")):
        if digest(json.loads((data/filename).read_text()))!=summary["trajectory_rows_sha256" if name=="trajectory" else "answer_rows_sha256"]:
            raise ValueError("audited rows changed")
    manifest=json.loads(Path(model_manifest).read_text());snapshot=Path(manifest["snapshot_path"])
    if manifest.get("repository_id")!="Qwen/Qwen2.5-Omni-7B" or manifest.get("all_lfs_verified") is not True:
        raise ValueError("verified model manifest required")
    out=Path(out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    base=yaml.safe_load((ROOT/"research/configs/native_omni_lora.yaml").read_text())
    commands=[]
    for name in ("trajectory","answer_only"):
        config=deepcopy(base);config.update({"model_name_or_path":str(snapshot),"dataset_dir":str(data),
                 "dataset":"native_omni_"+name,"output_dir":str(out/(name+"-adapter")),
                 "use_audio_in_video":summary["use_audio_in_video"],"cutoff_len":audit["cutoff_len"]})
        path=out/("train-"+name+".yaml");path.write_text(yaml.safe_dump(config,sort_keys=False))
        commands.append(f"PYTHONPATH=SFT/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/local/bin/pixi run --locked --manifest-path runtime/sft-gpu/pixi.toml python SFT/src/train.py {path}")
    (out/"commands.txt").write_text("\n".join(commands)+"\n")
    write_json(out/"preparation.json",{"source_split":"train","retained_pool_sha256":summary["retained_pool_sha256"],
                                     "native_trainer":"SFT/src/train.py","model_manifest_sha256":file_sha(model_manifest),
                                     "training_executed":False,"commands":commands})
    return commands


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    export=sub.add_parser("export");export.add_argument("--episodes-dir",required=True);export.add_argument("--out-dir",required=True)
    export.add_argument("--verdicts");export.add_argument("--audit-development",action="store_true")
    export.add_argument("--case-id",action="append")
    audit=sub.add_parser("audit");audit.add_argument("--dataset-dir",required=True);audit.add_argument("--model-manifest",required=True)
    audit.add_argument("--out-dir",required=True);audit.add_argument("--cutoff-len",type=int,default=32768)
    config=sub.add_parser("config");config.add_argument("--dataset-dir",required=True);config.add_argument("--audit-dir",required=True)
    config.add_argument("--model-manifest",required=True);config.add_argument("--out-dir",required=True)
    args=parser.parse_args()
    if args.command=="export":
        verdicts=json.loads(Path(args.verdicts).read_text()) if args.verdicts else None
        result=export_episodes(args.episodes_dir,args.out_dir,verdicts=verdicts,audit_only=args.audit_development,case_ids=args.case_id)
    elif args.command=="audit":
        result=audit_dataset(args.dataset_dir,args.model_manifest,args.out_dir,cutoff_len=args.cutoff_len)
        result={k:v for k,v in result.items() if k!="rows"}
    else:
        result=materialize_configs(args.dataset_dir,args.audit_dir,args.model_manifest,args.out_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
