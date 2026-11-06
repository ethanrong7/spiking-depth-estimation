"""Fresh repeat of the selected initial-screen O/I contrast; no scientific edit."""
import hashlib,sys
from pathlib import Path
base=Path(__file__).resolve().parents[2]/'alif-mechanism-screen-v1/scripts'
expected={'mechanism.py': 'f8a4147f682a4794ee7f814defcb3684d14764958d61b6606e527de9865f9ae0', 'run_screen.py': '7247b190d3db59274600917b407e14b22cc7fcc2512115e2fb250cb9069605ca'}
for name,digest in expected.items():
 if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('initialscientificimplementation changed: '+name)
sys.path.insert(0,str(base))
import run_screen
if __name__=='__main__':run_screen.driver.main()
