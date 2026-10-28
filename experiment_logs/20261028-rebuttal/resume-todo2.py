"""Resume only the timing stage after fixing evaluator self-matching."""
import json
import os
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.llama_cpp_common import collect_environment,utc_now,write_json_atomic
from scripts.run_llama_cpp_ffn_breakdown import capture
from scripts.run_llama_cpp_qpu_ffn_island_eval import git_record,require_no_competing_workloads

path=OUT/'todo2/screen-attempt.json'
r=json.loads(path.read_text())
assert r['numerically_accepted_fractions']==[0.0625,0.125,0.1875]
assert all(x['returncode']==0 and not x['invalid_reason'] for x in r['preflights'])
assert not (OUT/'todo2/screen.json').exists()
r['execution_attempts']=[r['execution']]
r['timing_source']=git_record(ROOT)
r['timing_resumed_utc']=utc_now()
def save():write_json_atomic(path,r)
def sudo(*args,value=None):subprocess.run(['sudo','-n',*args],input=value,text=True,check=True,stdout=subprocess.DEVNULL)
policies={p:p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')}
swaps=[x.split() for x in Path('/proc/swaps').read_text().splitlines()[1:]]
env={k:v for k,v in os.environ.items() if not k.startswith('GGML_QPU_') and k not in ('LD_PRELOAD','LD_LIBRARY_PATH')}
env.update({'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'})
save()
try:
    require_no_competing_workloads('before resumed timing')
    for p in policies:sudo('tee',str(p),value='performance\n')
    sudo('systemctl','mask','--runtime','dev-zram0.swap')
    for swap in swaps:sudo('swapoff',swap[0])
    r['timing_environment_before']=collect_environment();save()
    result=capture(r['screen_command'],env,OUT/'todo2/screen-retry',timeout_seconds=3600)
    r['execution']=result
    r['completed']=result['returncode']==0 and not result['invalid_reason']
    save()
finally:
    for p,g in policies.items():sudo('tee',str(p),value=g+'\n')
    sudo('systemctl','unmask','--runtime','dev-zram0.swap')
    subprocess.run(['sudo','-n','systemctl','reset-failed','dev-zram0.swap','systemd-zram-setup@zram0.service'],check=False)
    if any(x[0]=='/dev/zram0' for x in swaps):sudo('systemctl','start','dev-zram0.swap')
    r['timing_environment_after']=collect_environment();r['finished_utc']=utc_now()
    r['power_clean']=not any(x.get('undervoltage_event_count',0) for x in r['preflights']+[r.get('execution',{})]);save()
print(json.dumps({'completed':r.get('completed'),'returncode':r.get('execution',{}).get('returncode')},indent=2))
