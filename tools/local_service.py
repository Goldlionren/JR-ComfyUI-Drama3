"""Start, inspect, and stop the current user's loopback Drama 3 teacher provider."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import psutil

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".runtime/provider-service"
LOCAL_HTTP = build_opener(ProxyHandler({}))


@contextmanager
def service_lock(name):
    with (STATE / name).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_record(name):
    file = STATE / (name + ".json")
    if not file.exists():
        return None
    record = json.loads(file.read_text("utf-8"))
    try:
        process = psutil.Process(record["pid"])
        if (process.create_time() == record["created"] and
                process.cmdline() == record["command"] and process.is_running()):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def local_port(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def start(name, args, env, port, ready_path):
    current = read_record(name)
    if current is None:
        if local_port(port):
            raise RuntimeError(f"Port {port} is already occupied by a different process.")
        with (STATE / (name + ".log")).open("ab", buffering=0) as log:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt")
        current = psutil.Process(child.pid)
        record = dict(pid=child.pid, created=current.create_time(), command=current.cmdline())
        tmp = STATE / (name + ".tmp")
        tmp.write_text(json.dumps(record), "utf-8")
        tmp.replace(STATE / (name + ".json"))
    until = time.monotonic() + 150
    while time.monotonic() < until:
        if not current.is_running():
            break
        try:
            with LOCAL_HTTP.open(f"http://127.0.0.1:{port}{ready_path}", timeout=4) as response:
                if response.status == 200:
                    print(f"{name} ready (port {port}, PID {current.pid})", flush=True)
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.6)
    raise RuntimeError(f"{name} did not become ready. See {STATE / (name + '.log')}")


def stop(name):
    process = read_record(name)
    if process:
        process.terminate()
        process.wait(timeout=20)
        print(f"Stopped {name} (PID {process.pid})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--aki", type=Path, default=Path("F:/ComfyUI-aki-v3"))
    parser.add_argument("--comfy-root", type=Path,
                        help="Directory containing main.py. Use this ComfyUI environment's Python to run the launcher.")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        for name in ("provider", "worker"):
            process = read_record(name)
            print(name, f"PID {process.pid}" if process else "stopped")
        return
    with service_lock("service.lock"):
        if args.action == "stop":
            stop("provider")
            stop("worker")
            return
        if args.comfy_root:
            comfy = args.comfy_root.resolve(strict=True)
            python = sys.executable
        elif os.name == "nt":
            comfy = (args.aki / "ComfyUI").resolve()
            python = (args.aki / "python/python.exe").resolve(strict=True)
        else:
            parser.error("Supply --comfy-root /path/to/ComfyUI when starting the teacher service.")
        if not (comfy / "custom_nodes/JR-ComfyUI-Drama3").is_dir():
            raise FileNotFoundError(f"Install the plugin at {comfy / 'custom_nodes/JR-ComfyUI-Drama3'} first.")
        worker = STATE / "comfy"
        for folder in ("user", "temp", "output", "input"):
            (worker / folder).mkdir(parents=True, exist_ok=True)
        voice_db = Path(os.environ.get("JR_DRAMA3_VOICE_DB", comfy / "user/jr_drama3/voices.sqlite3")).absolute()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", NUMBA_CACHE_LOCATOR_CLASSES="UserProvidedCacheLocator",
            NUMBA_CACHE_DIR=str(ROOT/".runtime/numba"), TRITON_CACHE_DIR=str(ROOT/".runtime/triton"),
            TORCHINDUCTOR_CACHE_DIR=str(ROOT/".runtime/inductor"), JR_DRAMA3_VOICE_DB=str(voice_db))
        worker_cmd = [str(python), "-B", "-u", str(comfy/"main.py"), "--base-directory", str(comfy),
            "--user-directory", str(worker/"user"), "--temp-directory", str(worker/"temp"),
            "--input-directory", str(worker/"input"), "--output-directory", str(worker/"output"),
            "--offline", "--listen", "127.0.0.1", "--port", "8194", "--disable-assets",
            "--disable-auto-launch", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "JR-ComfyUI-Drama3"]
        start("worker", worker_cmd, env, 8194, "/object_info/JRDrama3Generate")
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
        cmd = [str(python), "-B", "-u", "-m", "provider", "--port", "8193",
               "--comfy-url", "http://127.0.0.1:8188", "--fallback-comfy-url", "http://127.0.0.1:8194",
               "--voice-db", str(voice_db), "--ffmpeg", ffmpeg]
        start("provider", cmd, env, 8193, "/health")


if __name__ == "__main__":
    main()
