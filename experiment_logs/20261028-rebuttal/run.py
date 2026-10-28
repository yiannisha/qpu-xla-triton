"""Execute the coa.md workflow sequentially with an incremental stage record."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.llama_cpp_common import collect_environment, utc_now, write_json_atomic
from scripts.run_llama_cpp_ffn_breakdown import capture

LLAMA = Path('/home/yiannis/side/llama.cpp')
MODEL = Path('/home/yiannis/side/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf')
PROFILE = ROOT / 'build/llama-qpu-profile'
CPU = ROOT / 'build/ffn-profile-cpu'
os.chdir(ROOT)
env = {k:v for k,v in os.environ.items() if not k.startswith('GGML_QPU_') and k not in ('LD_PRELOAD','LD_LIBRARY_PATH')}
env.update({'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
record={'started_utc':utc_now(),'environment_original':collect_environment(),'stages':[]}

def save():
    write_json_atomic(OUT/'campaign.json',record)
    with (OUT/'campaign.json').open('rb') as f: os.fsync(f.fileno())

def stage(name,command,environment=None,native=False):
    item={'name':name,'command':list(map(str,command)),'started_utc':utc_now()}
    record['stages'].append(item);save()
    print(f"{utc_now()} starting {name}",flush=True)
    if native:
        result=capture(item['command'],environment or env,OUT/name,timeout_seconds=300)
        item.update(result)
    else:
        with (OUT/f'{name}.log').open('w') as log:
            process=subprocess.run(item['command'],cwd=ROOT,env=environment or env,
                                   stdout=log,stderr=subprocess.STDOUT,check=False)
            log.flush();os.fsync(log.fileno())
        item['returncode']=process.returncode
    item['finished_utc']=utc_now();save()
    print(f"{utc_now()} finished {name}: {item['returncode']}",flush=True)
    if item['returncode'] or item.get('invalid_reason'):
        raise RuntimeError(f"stage {name} failed; inspect stage logs")

try:
    stage('historical-audit',[sys.executable,'scripts/audit_llama_cpp_ffn_calibration.py','--output',OUT/'historical-calibration.json'])
    stage('stride',[PROFILE/'qpu_repack_stride_smoke'],native=True)
    stage('joined-graph',[PROFILE/'qpu_ffn_island_graph_smoke'],
          {**env,'LD_LIBRARY_PATH':str(CPU),'GGML_QPU_FFN_PROFILE':'1'},native=True)
    stage('full-vocabulary',[PROFILE/'qpu_ffn_island_model_smoke',MODEL,'0.125'],
          {**env,'LD_LIBRARY_PATH':str(CPU)},native=True)
    stage('todo1',[sys.executable,'-u','scripts/run_llama_cpp_ffn_breakdown.py',
          '--llama-root',LLAMA,'--model',MODEL,'--pairs','5','--output',OUT/'todo1'])
    stage('todo1-summary',[sys.executable,'scripts/summarize_llama_cpp_ffn_breakdown.py',
          OUT/'todo1/breakdown.json','--output',OUT/'todo1-summary.json'])
    stage('todo2',[sys.executable,'-u','scripts/run_llama_cpp_ffn_split_screen.py',
          '--llama-root',LLAMA,'--model',MODEL,'--repetitions','3','--output',OUT/'todo2'])
    stage('todo2-summary',[sys.executable,'scripts/summarize_llama_cpp_ffn_split_screen.py',
          OUT/'todo2','--output',OUT/'todo2-grid.json'])
    record['completed']=True
except Exception as error:
    record['completed']=False;record['error']=str(error)
    print(str(error),flush=True)
finally:
    record['finished_utc']=utc_now();record['environment_after']=collect_environment();save()
