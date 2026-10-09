"""Bounded native 8B inference and local-tool smoke; not benchmark accuracy."""
from pathlib import Path
import functools,hashlib,json,logging,os,re,subprocess,sys,threading,time
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'repro'/'assets'
MODEL=ASSETS/'model-8b'
# The completed downloader receipt is required before any weight load.
receipt=json.loads((MODEL/'download-receipt.json').read_text())
if not receipt.get('verified') or receipt['sha256']!='f73186b544f7ff267d1150cc5ef3340edbf196d21b46794e4a623734304c8c5b':
 raise RuntimeError('Verified published weight receipt required')
for key,value in {'ONESEARCH_DISABLE_RETRIEVAL':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_ATTN_IMPLEMENTATION':'sdpa','AGENT_MAX_TURNS':'3','SINGLE_IMAGE_MAX_TURNS':'3','VIDEO_AGENT_MAX_TURNS':'3','VIDEO_MAX_FRAMES':'4','VIDEO_MAX_PIXELS':'50176','VIDEO_FPS':'1'}.items():os.environ[key]=value
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
stamp=time.strftime('%Y%m%d-%H%M%S',time.gmtime())
OUT=ROOT/'repro'/'results'/('minimal-'+stamp)
OUT.mkdir(parents=True,exist_ok=False)
class QuietHandler(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(QuietHandler,directory=str(ASSETS)))
threading.Thread(target=server.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{server.server_port}'
os.environ['ASSET_DIR']=str(ASSETS/'published');os.environ['ASSET_BASE_URL']=base+'/published'
sys.path.insert(0,str(ROOT/'onesearch_vl'))
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
import torch
from PIL import Image,ImageDraw
from onesearch_infer.runners import build_runner,InferenceConfig
from onesearch_infer import tools,video_io
from onesearch_infer.pipeline_single import process_single_case_single
from onesearch_infer.pipeline import process_single_case
threads=min(16,os.cpu_count() or 1);torch.set_num_threads(threads)
subprocess.run([sys.executable,str(ASSETS/'video-subset'/'prepare.py'),'--data-root',str(ASSETS/'video-subset'),'--media-base-url',base+'/video-subset'],check=True)
manifest=json.loads((ASSETS/'video-subset'/'runtime'/'video_manifest.jsonl').read_text().splitlines()[0])
grid=video_io.build_grid_from_manifest(manifest)
handles={'video_1':grid}
intermediate=OUT/'manual-local-tools';intermediate.mkdir()
tool_results=[]
for turn,call in enumerate([
 {'name':'select_frame','parameters':{'video':'video_1','frame_index':0}},
 {'name':'select_timespan','parameters':{'video':'video_1','start_index':0,'end_index':4}},
 {'name':'crop','parameters':{'image':'img_1','x':0,'y':0,'width':128,'height':128}},
]):
 message,new=tools.execute_tool(call,handles,'manual-local-smoke',0,turn,str(intermediate),modality='video')
 if message.startswith('Tool execution error') or not new:raise RuntimeError(message)
 handles.update(new);tool_results.append({'call':call,'observation':message,'new_handles':list(new)})
(OUT/'local-tools.json').write_text(json.dumps(tool_results,indent=2))
print('Verified real local select_frame/select_timespan/crop',flush=True)
t0=time.monotonic()
runner=build_runner('8b',checkpoint=str(MODEL),gpus='0',dtype='bfloat16')
runner.load()
load_seconds=time.monotonic()-t0
print('Published 8B loaded on',runner._device,'in',load_seconds,'seconds',flush=True)
image=ASSETS/'synthetic-smoke.png'
im=Image.new('RGB',(256,256),'white');ImageDraw.Draw(im).rectangle((64,64,192,192),fill='red');im.save(image)
cfg=InferenceConfig(temperature=0,max_tokens=256)
row={'data_id':'synthetic-red-square','question':'Read img_1 directly and name the color and shape of its single colored object. Do not call tools. Keep reasoning brief. Finish with <response>color shape</response>.','images':[str(image)]}
t0=time.monotonic()
single=process_single_case_single(row,runner,str(OUT/'single-image'),0,inference_cfg=cfg)
single_seconds=time.monotonic()-t0
# This custom question uses unchanged released video/PNG assets. It is a pixel/tool smoke.
video_input_manifest=dict(manifest)
video_relative=json.loads((ASSETS/'video-subset'/'test.jsonl').read_text())['video']
video_input_manifest['video_url']=str(ASSETS/'video-subset'/video_relative)
video_row={'data_id':'published-video-local-inspection','question':'Use only select_frame with video=video_1 and frame_index=127 once, then read the visible closing-screen text directly from its image. layout_parsing and all external APIs are unavailable; do not call any other tools. Keep reasoning brief and finish with <response>text</response>.'}
t0=time.monotonic()
video=process_single_case(video_row,runner,str(OUT/'video'),0,manifest_record=video_input_manifest,inference_cfg=InferenceConfig(temperature=0,max_tokens=256))
video_seconds=time.monotonic()-t0
for label,traj in [('single_image',single),('video',video)]:
 if traj.get('status')=='error' or not traj.get('turns') or any(t.get('error') for t in traj['turns']):raise RuntimeError(f'{label} inference failed')
 output=traj.get('final_response_text') or ''
 if not re.search(r'<response>.*?</response>',output,re.S):raise RuntimeError(f'{label} missing completed response block')
 for turn in traj['turns']:
  if 'Tool execution error' in (turn.get('tool_output') or ''):raise RuntimeError(f'{label} tool execution failed')
  call=turn.get('response_text') or turn.get('tool_call')
  if call and any(x in str(call) for x in ('text_search','image_search','web_search')):raise RuntimeError('Actor emitted prohibited retrieval')
meta={'scope':'published 8B + native image/video agent + real local tools; retrieval disabled; custom questions; no benchmark score','git_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model_revision':receipt['revision'],'weight_sha256':receipt['sha256'],'dataset_revision':'8694a570d777066e053fc258623a429cc13a2294','one_search_disable_retrieval':True,'device':str(runner._device),'dtype':str(next(runner._model.parameters()).dtype),'model_config_use_cache':runner._model.config.text_config.use_cache,'torch_threads':threads,'video_transport':'local path to unchanged released MP4; PNGs and local tools via loopback HTTP','video_max_frames':4,'video_max_pixels':50176,'single_max_tokens_per_turn':256,'video_max_tokens_per_turn':256,'max_turns':3,'load_seconds':load_seconds,'single_seconds':single_seconds,'video_seconds':video_seconds,'single_status':single.get('status'),'video_status':video.get('status'),'single_final':single.get('final_response_text',''),'video_final':video.get('final_response_text',''),'output_dir':str(OUT)}
(OUT/'run-manifest.json').write_text(json.dumps(meta,indent=2))
(ROOT/'repro'/'results'/'latest-minimal.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
server.shutdown();server.server_close()
