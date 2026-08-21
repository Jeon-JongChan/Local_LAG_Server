"""
llama-server 자동 준비 (Windows / Linux).

1) LLAMA_CPP_BASE_URL 포트가 응답하면 종료
2) tools/llama-cpp 에 llama-server 없으면 GitHub 릴리스 다운로드·압축 해제
   - Windows: *-bin-win-*.zip
   - Linux:   *-bin-ubuntu-*.tar.gz (우선) / .zip
3) data/models/chat 에 GGUF 없으면 Hugging Face에서 다운로드
4) llama-server 기동 후 헬스체크
   - Windows: 새 콘솔 창 / Linux: 백그라운드 프로세스

환경 변수 (local-rag.bat / local-rag.sh 에서 export):
  LLAMA_CPP_BASE_URL, LLAMA_CPP_VARIANT, LLAMA_GGUF_REPO, LLAMA_GGUF_FILE,
  LLAMA_GGUF_URL (선택, 직접 URL), LLAMA_CTX_SIZE, LLAMA_NGL, LLAMA_CPP_RELEASE_TAG
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# scripts/ → 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from settings import CHAT_MODELS_DIR  # noqa: E402

TOOLS_DIR = ROOT / "tools" / "llama-cpp"
META_PATH = TOOLS_DIR / ".install_meta.json"

USER_AGENT = "local-rag-ensure-llama/1.0"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/{tag}"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _parse_base_url() -> tuple[str, int]:
    base = _env("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1").rstrip("/")
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return host, port


def _port_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def server_healthy(host: str, port: int) -> bool:
    """포트 LISTEN + HTTP 헬스(/health 또는 /v1/models)."""
    if not _port_listening(host, port):
        return False
    for path in ("/health", "/v1/models", "/"):
        if _http_ok(f"http://{host}:{port}{path}"):
            return True
    return False


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[llama] 다운로드: {label}")
    print(f"        {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        chunk = 1024 * 1024
        with tmp.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total > 0 and done % (10 * chunk) < chunk:
                    pct = done * 100 // total
                    print(f"        ... {done // (1024*1024)} MB ({pct}%)")
    tmp.replace(dest)
    print(f"[llama] 저장: {dest}")


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cpu_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def _detect_variant() -> str:
    explicit = _env("LLAMA_CPP_VARIANT").lower()
    if explicit in ("cpu", "vulkan", "cuda"):
        return explicit
    # NVIDIA 있으면: Windows=cuda, Linux=공식 CUDA 빌드 없음 → vulkan
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if r.returncode == 0:
            return "cuda" if sys.platform == "win32" else "vulkan"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "vulkan" if sys.platform == "win32" else "cpu"


def _pick_asset_url(release: dict, variant: str) -> Optional[str]:
    assets = release.get("assets") or []
    arch = _cpu_arch()

    def find_win(substr: str) -> Optional[str]:
        for a in assets:
            n = a.get("name", "")
            if substr in n and n.endswith(".zip") and "win" in n and arch in n:
                return a.get("browser_download_url")
        return None

    def find_linux(substr: str) -> Optional[str]:
        # tar.gz 우선, 없으면 zip
        for ext in (".tar.gz", ".zip"):
            for a in assets:
                n = a.get("name", "")
                if substr in n and n.endswith(ext) and "ubuntu" in n:
                    return a.get("browser_download_url")
        return None

    if sys.platform == "win32":
        if variant == "cpu":
            url = find_win(f"bin-win-cpu-{arch}")
            if url:
                return url
        if variant == "vulkan":
            url = find_win(f"bin-win-vulkan-{arch}")
            if url:
                return url
        if variant == "cuda":
            for cuda_tag in (
                f"bin-win-cuda-13.3-{arch}",
                f"bin-win-cuda-12.4-{arch}",
                f"bin-win-cuda-{arch}",
                "bin-win-cuda",
            ):
                url = find_win(cuda_tag)
                if url:
                    return url
        return find_win(f"bin-win-cpu-{arch}")

    # Linux (ubuntu 프리빌드). 공식 릴리스에 CUDA ubuntu 빌드는 없음.
    if variant == "cuda":
        print(
            "[llama] Linux 공식 릴리스에 CUDA 빌드가 없어 vulkan/cpu 로 대체합니다. "
            "(LLAMA_CPP_VARIANT=vulkan|cpu)"
        )
        variant = "vulkan"
    if variant == "vulkan":
        url = find_linux(f"bin-ubuntu-vulkan-{arch}")
        if url:
            return url
    # cpu (또는 vulkan 폴백)
    url = find_linux(f"bin-ubuntu-{arch}")
    if url and "vulkan" not in Path(urlparse(url).path).name:
        return url
    # 이름에 vulkan 없는 ubuntu-{arch} 를 직접 고르기
    for a in assets:
        n = a.get("name", "")
        if f"bin-ubuntu-{arch}" in n and "vulkan" not in n and "rocm" not in n:
            if n.endswith(".tar.gz") or n.endswith(".zip"):
                return a.get("browser_download_url")
    return find_linux(f"bin-ubuntu-{arch}")


def _get_release_download_url(variant: str) -> tuple[str, str]:
    tag = _env("LLAMA_CPP_RELEASE_TAG")
    if tag:
        if not tag.startswith("b"):
            tag = f"b{tag}"
        api_url = GITHUB_API.format(tag=f"tags/{tag}")
    else:
        api_url = GITHUB_API.format(tag="latest")

    release = _fetch_json(api_url)
    version = release.get("tag_name", "unknown")
    url = _pick_asset_url(release, variant)
    if not url:
        os_label = "Windows" if sys.platform == "win32" else "Linux(ubuntu)"
        raise RuntimeError(
            f"GitHub 릴리스 {version} 에서 variant={variant} {os_label} 패키지를 찾지 못했습니다. "
            "LLAMA_CPP_VARIANT=cpu|vulkan|cuda 를 바꿔 보세요."
        )
    return url, version


def _find_llama_server_exe() -> Optional[Path]:
    if not TOOLS_DIR.is_dir():
        return None
    names = ("llama-server.exe", "llama-server")
    for name in names:
        for p in TOOLS_DIR.rglob(name):
            if p.is_file():
                return p
    return None


def _extract_archive(archive: Path, dest: Path) -> None:
    print(f"[llama] 압축 해제: {dest}")
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
    elif archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    else:
        raise RuntimeError(f"지원하지 않는 압축 형식: {archive.name}")


def _install_llama_cpp(variant: str) -> Path:
    existing = _find_llama_server_exe()
    if existing:
        return existing

    url, version = _get_release_download_url(variant)
    suffix = ".tar.gz" if url.endswith(".tar.gz") else ".zip"
    archive_path = TOOLS_DIR / f"llama-{version}-{variant}{suffix}"
    _download(url, archive_path, f"llama.cpp {version} ({variant})")

    extract_to = TOOLS_DIR / version
    if extract_to.is_dir():
        shutil.rmtree(extract_to, ignore_errors=True)
    extract_to.mkdir(parents=True, exist_ok=True)

    _extract_archive(archive_path, extract_to)

    try:
        archive_path.unlink()
    except OSError:
        pass

    META_PATH.write_text(
        json.dumps(
            {
                "version": version,
                "variant": variant,
                "platform": sys.platform,
                "arch": _cpu_arch(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exe = _find_llama_server_exe()
    if not exe:
        raise RuntimeError("압축 해제 후 llama-server 를 찾지 못했습니다.")
    if sys.platform != "win32":
        try:
            exe.chmod(exe.stat().st_mode | 0o111)
        except OSError:
            pass
    return exe


def _ensure_gguf_model() -> Path:
    """
    GGUF 채팅 모델을 data/models/chat/ 에 준비합니다.

    우선순위:
      1) LLAMA_GGUF_REPO + LLAMA_GGUF_FILE → huggingface_hub (권장, 단일 파일)
      2) LLAMA_GGUF_URL → 직접 HTTP (단일 파일 URL 만; 분할 gguf 는 404)

    공식 Qwen/Qwen2.5-*-GGUF 의 q4_k_m 은 *-00001-of-00002.gguf 처럼
    분할만 있어 예전 단일 URL 은 404 가 납니다.
    """
    CHAT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    repo = _env("LLAMA_GGUF_REPO", "bartowski/Qwen_Qwen3.5-9B-GGUF")
    filename = _env("LLAMA_GGUF_FILE", "Qwen_Qwen3.5-9B-Q4_K_M.gguf")
    url_override = _env("LLAMA_GGUF_URL", "")

    if url_override and not _env("LLAMA_GGUF_REPO"):
        filename = _env("LLAMA_GGUF_FILENAME", "") or url_override.split("/")[-1].split("?")[0]
        dest = CHAT_MODELS_DIR / filename
    else:
        dest = CHAT_MODELS_DIR / filename

    # 예전 경로 data/models/*.gguf 호환
    legacy = ROOT / "data" / "models" / filename

    min_bytes = 100 * 1024 * 1024  # 100MB 미만이면 깨진/부분 다운로드로 간주
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        print(f"[llama] GGUF 이미 있음: {dest}")
        return dest
    if legacy.is_file() and legacy.stat().st_size >= min_bytes:
        print(f"[llama] GGUF (legacy 경로): {legacy}")
        return legacy
    if dest.is_file() and dest.stat().st_size < min_bytes:
        print(f"[llama] 불완전한 파일 삭제 후 재다운로드: {dest}")
        try:
            dest.unlink()
        except OSError:
            pass
    part = dest.with_suffix(dest.suffix + ".part")
    if part.is_file():
        try:
            part.unlink()
        except OSError:
            pass

    if url_override and not _env("LLAMA_GGUF_REPO"):
        _download(url_override, dest, f"GGUF {filename}")
        return dest

    print(f"[llama] Hugging Face: {repo} / {filename}")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(CHAT_MODELS_DIR),
        )
        return Path(path)
    except ImportError:
        print("[llama] huggingface_hub 없음 — pip install huggingface_hub 후 재시도 권장")
        resolve = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        _download(resolve, dest, f"GGUF {filename}")
        return dest


def _configured_gguf_label() -> str:
    name = _env("LLAMA_GGUF_FILE", "")
    repo = _env("LLAMA_GGUF_REPO", "")
    if name and repo:
        return f"{name}  ({repo})"
    if name:
        return name
    url = _env("LLAMA_GGUF_URL", "")
    if url:
        return url.rsplit("/", 1)[-1]
    return "(LLAMA_GGUF_FILE 미설정)"


def _probe_loaded_model(host: str, port: int) -> str:
    """실행 중 llama-server 의 /v1/models 첫 id (없으면 빈 문자열)."""
    try:
        req = urllib.request.Request(
            f"http://{host}:{port}/v1/models",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        items = data.get("data") or []
        if not items:
            return ""
        first = items[0] if isinstance(items[0], dict) else {}
        return str(first.get("id") or first.get("model") or "").strip()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError):
        return ""


def _print_chat_model_status(host: str, port: int, *, running: bool) -> None:
    print(f"[llama] 채팅 GGUF(설정): {_configured_gguf_label()}")
    if running:
        loaded = _probe_loaded_model(host, port)
        if loaded:
            print(f"[llama] 서버 로드 모델: {loaded}")
        else:
            print("[llama] 서버 로드 모델: (조회 실패 — /v1/models)")


def _start_server(exe: Path, model: Path, host: str, port: int) -> None:
    ctx = _env("LLAMA_CTX_SIZE", "4096")
    variant = _env("LLAMA_CPP_VARIANT") or _detect_variant()
    ngl = _env("LLAMA_NGL", "")
    if not ngl:
        ngl = "0" if variant == "cpu" else "999"

    bind = "127.0.0.1" if host in ("0.0.0.0", "::", "127.0.0.1") else host
    cmd = [
        str(exe),
        "-m",
        str(model),
        "--host",
        bind,
        "--port",
        str(port),
        "-c",
        ctx,
        "-ngl",
        ngl,
    ]

    # CPU 스레드 (생성 / 프롬프트·배치 처리)
    threads = _env("LLAMA_THREADS", "")
    if threads:
        cmd += ["-t", threads]
    threads_batch = _env("LLAMA_THREADS_BATCH", "")
    if threads_batch:
        cmd += ["-tb", threads_batch]

    # Flash Attention: on | off (비우면 llama.cpp 기본값)
    flash_attn = _env("LLAMA_FLASH_ATTN", "")
    if flash_attn:
        cmd += ["-fa", flash_attn]

    # 배치 크기 (프롬프트 처리 성능·메모리 트레이드오프)
    batch = _env("LLAMA_BATCH_SIZE", "")
    if batch:
        cmd += ["-b", batch]
    ubatch = _env("LLAMA_UBATCH_SIZE", "")
    if ubatch:
        cmd += ["-ub", ubatch]

    # Qwen thinking 제어: jinja 템플릿 + 요청별 chat_template_kwargs
    # --reasoning auto → 클라이언트 enable_thinking 으로 instant/think 전환
    # 전역 OFF: LLAMA_REASONING=off 또는 LLAMA_REASONING_BUDGET=0
    cmd += ["--jinja"]
    reasoning = (_env("LLAMA_REASONING", "auto") or "auto").lower()
    if reasoning in ("on", "off", "auto"):
        cmd += ["--reasoning", reasoning]
    budget = _env("LLAMA_REASONING_BUDGET", "")
    if budget != "":
        cmd += ["--reasoning-budget", budget]

    print(f"[llama] 채팅 GGUF: {model.name}")
    print("[llama] 서버 시작:")
    print("        " + " ".join(cmd))

    creationflags = 0
    stdout = None
    stderr = None
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_CONSOLE
    else:
        log_path = TOOLS_DIR / "llama-server.log"
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")
        stdout = log_f
        stderr = subprocess.STDOUT
        print(f"[llama] 로그: {log_path}")

    subprocess.Popen(
        cmd,
        cwd=str(exe.parent),
        creationflags=creationflags,
        stdout=stdout,
        stderr=stderr,
        start_new_session=(sys.platform != "win32"),
    )


def _wait_healthy(host: str, port: int, seconds: int = 120) -> bool:
    print(f"[llama] 서버 기동 대기 (최대 {seconds}초)...")
    deadline = time.time() + seconds
    while time.time() < deadline:
        if server_healthy(host, port):
            print(f"[llama] 준비 완료: http://{host}:{port}")
            _print_chat_model_status(host, port, running=True)
            return True
        time.sleep(2)
    return False


def main() -> int:
    if sys.platform not in ("win32", "linux"):
        print(
            f"[llama] 자동 설치는 Windows/Linux 만 지원합니다 (현재: {sys.platform}). "
            "수동으로 llama-server 를 실행하세요."
        )
        host, port = _parse_base_url()
        if server_healthy(host, port):
            _print_chat_model_status(host, port, running=True)
            return 0
        return 1

    host, port = _parse_base_url()

    if server_healthy(host, port):
        print(f"[llama] 이미 실행 중: http://{host}:{port}")
        _print_chat_model_status(host, port, running=True)
        return 0

    print("[llama] llama-server 미응답 — 설치·모델·기동을 시도합니다.")
    print(f"[llama] 채팅 GGUF(설정): {_configured_gguf_label()}")

    try:
        variant = _detect_variant()
        print(f"[llama] 빌드 variant: {variant} (LLAMA_CPP_VARIANT 로 변경 가능)")
        exe = _install_llama_cpp(variant)
        model = _ensure_gguf_model()
        _start_server(exe, model, host, port)
    except Exception as e:
        print(f"[오류] llama 준비 실패: {e}", file=sys.stderr)
        return 1

    if _wait_healthy(host, port):
        return 0

    print(
        "[오류] llama-server 가 시간 내에 응답하지 않습니다. "
        "llama-server 로그를 확인하세요"
        + (
            f" ({TOOLS_DIR / 'llama-server.log'})"
            if sys.platform != "win32"
            else "."
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
