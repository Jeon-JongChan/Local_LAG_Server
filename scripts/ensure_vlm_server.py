"""
라벨러용 VL(멀티모달) llama-server 자동 준비.

1) config.py 의 VLM_PORT 가 응답하면 종료
2) tools/llama-cpp 에 llama-server 없으면 ensure_llama_server 설치 로직 재사용
3) llama-server -hf <VLM_HF_REPO> --port <VLM_PORT> 기동 후 헬스체크
   - Windows: 새 콘솔 창 / Linux: 백그라운드 + 로그

채팅용 :8080 서버와는 별개 프로세스입니다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
LABELER_DIR = ROOT / "tools" / "labeler"
sys.path.insert(0, str(LABELER_DIR))

from config import VLM_BASE_URL, VLM_HF_REPO, VLM_PORT, vlm_server_cmd_hint  # noqa: E402


def _load_ensure_llama():
    path = ROOT / "scripts" / "ensure_llama_server.py"
    spec = importlib.util.spec_from_file_location("ensure_llama_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"로드 실패: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_vlm_host_port() -> tuple[str, int]:
    parsed = urlparse(VLM_BASE_URL.rstrip("/") + "/")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or VLM_PORT
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return host, port


def _start_vlm_server(els, exe: Path, host: str, port: int) -> None:
    bind = "127.0.0.1" if host in ("0.0.0.0", "::", "127.0.0.1") else host
    variant = els._env("LLAMA_CPP_VARIANT") or els._detect_variant()
    ngl = els._env("LLAMA_NGL", "")
    if not ngl:
        ngl = "0" if variant == "cpu" else "999"

    cmd = [
        str(exe),
        "-hf",
        VLM_HF_REPO,
        "--host",
        bind,
        "--port",
        str(port),
        "-ngl",
        ngl,
    ]

    threads = els._env("LLAMA_THREADS", "")
    if threads:
        cmd += ["-t", threads]
    threads_batch = els._env("LLAMA_THREADS_BATCH", "")
    if threads_batch:
        cmd += ["-tb", threads_batch]

    print(f"[vlm] 모델(-hf): {VLM_HF_REPO}")
    print("[vlm] 서버 시작:")
    print("      " + " ".join(cmd))

    creationflags = 0
    stdout = None
    stderr = None
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        log_path = els.TOOLS_DIR / "llama-vlm-server.log"
        els.TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")
        stdout = log_f
        stderr = subprocess.STDOUT
        print(f"[vlm] 로그: {log_path}")

    subprocess.Popen(
        cmd,
        cwd=str(exe.parent),
        creationflags=creationflags,
        stdout=stdout,
        stderr=stderr,
        start_new_session=(sys.platform != "win32"),
    )


def _wait_healthy(els, host: str, port: int, seconds: int = 600) -> bool:
    print(f"[vlm] 서버 기동 대기 (최대 {seconds}초, 첫 실행은 모델 다운로드 포함)...")
    deadline = time.time() + seconds
    while time.time() < deadline:
        if els.server_healthy(host, port):
            print(f"[vlm] 준비 완료: http://{host}:{port}")
            return True
        time.sleep(3)
    return False


def main() -> int:
    if sys.platform not in ("win32", "linux"):
        print(
            f"[vlm] 자동 기동은 Windows/Linux 만 지원합니다 (현재: {sys.platform}). "
            f"수동 실행: {vlm_server_cmd_hint()}"
        )
        return 1

    els = _load_ensure_llama()
    host, port = _parse_vlm_host_port()

    if els.server_healthy(host, port):
        print(f"[vlm] 이미 실행 중: http://{host}:{port}")
        return 0

    print(f"[vlm] VL 서버 미응답 (port {port}) — 설치·기동을 시도합니다.")
    try:
        variant = els._detect_variant()
        print(f"[vlm] 빌드 variant: {variant}")
        exe = els._install_llama_cpp(variant)
        _start_vlm_server(els, exe, host, port)
    except Exception as e:
        print(f"[오류] VLM llama-server 준비 실패: {e}", file=sys.stderr)
        print(f"        수동 실행: {vlm_server_cmd_hint()}", file=sys.stderr)
        return 1

    if _wait_healthy(els, host, port):
        return 0

    log_hint = (
        f" ({els.TOOLS_DIR / 'llama-vlm-server.log'})"
        if sys.platform != "win32"
        else "."
    )
    print(
        "[오류] VL llama-server 가 시간 내에 응답하지 않습니다."
        f" 로그를 확인하세요{log_hint}",
        file=sys.stderr,
    )
    print(f"        수동 실행: {vlm_server_cmd_hint()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
