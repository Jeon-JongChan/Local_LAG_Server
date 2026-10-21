"""llama-server 바이너리 및 프로세스 관리자 어댑터 구현체.

GitHub llama.cpp 릴리스 바이너리 다운로드, GGUF 모델 준비 및 서버 프로세스 기동/헬스체크를 관리합니다.
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
from typing import Optional, Tuple
from urllib.parse import urlparse

from .base import BaseEngineManager

from settings import (
    BASE_DIR,
    CHAT_MODELS_DIR,
    DATA_DIR,
    LLAMA_CPP_BASE_URL,
    LLAMA_CPP_VARIANT,
    LLAMA_CTX_SIZE,
    LLAMA_GGUF_FILE,
    LLAMA_GGUF_REPO,
    LLAMA_MMAP,
    LLAMA_NGL,
    MODELS_DIR,
)

# 포터블 기본: data/serving_engines. 예전 tools/llama-cpp 에 바이너리 있으면 재사용.
_SERVING_DIR = DATA_DIR / "serving_engines" / "llama-cpp"
_LEGACY_TOOLS_DIR = BASE_DIR / "tools" / "llama-cpp"
_USER_AGENT = "local-rag/2.0 (llama-engine-manager)"


def _server_exe_name() -> str:
    """플랫폼별 llama-server 실행 파일명."""
    return "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _has_server(directory: Path) -> bool:
    """디렉터리에 llama-server 바이너리가 있는지 확인."""
    if not directory.is_dir():
        return False
    exe = _server_exe_name()
    if (directory / exe).is_file():
        return True
    return any(directory.rglob(exe))


def _default_tools_dir() -> Path:
    """신규 설치·다운로드는 항상 data/serving_engines/llama-cpp."""
    return _SERVING_DIR


class LlamaEngineManager(BaseEngineManager):
    """llama.cpp 바이너리 및 런타임 관리자."""

    def __init__(
        self,
        base_url: str = LLAMA_CPP_BASE_URL,
        tools_dir: Optional[Path] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tools_dir = tools_dir or _default_tools_dir()
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._proc: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        """포트 열림 및 프로세스 활성 여부 검사 (모델 준비 여부와 무관)."""
        if self._proc is not None and self._proc.poll() is not None:
            return False

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8080
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"

        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except (OSError, ConnectionRefusedError):
            return False

    def _get_health_status(self) -> Tuple[Optional[int], Optional[str]]:
        """llama-server /health 엔드포인트를 호출하여 HTTP 상태 코드와 본문 상태를 조회합니다.

        Returns:
            (status_code, body_status): 예: (200, "ok"), (503, "loading model"), (500, "error"), (None, None)
        """
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        port = parsed.port or 8080
        health_url = f"{scheme}://{host}:{port}/health"

        req = urllib.request.Request(health_url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                code = getattr(resp, "status", None) or getattr(resp, "code", 200)
                try:
                    raw_body = resp.read().decode("utf-8", errors="replace")
                    data = json.loads(raw_body)
                    if isinstance(data, dict):
                        status_text = data.get("status")
                    else:
                        status_text = raw_body.strip()
                except Exception:
                    status_text = None
                return code, str(status_text) if status_text is not None else None
        except urllib.error.HTTPError as e:
            code = e.code
            status_text = None
            try:
                raw_body = e.read().decode("utf-8", errors="replace")
                data = json.loads(raw_body)
                if isinstance(data, dict):
                    status_text = data.get("status")
                    if not status_text and "error" in data:
                        err = data.get("error")
                        status_text = err.get("message") if isinstance(err, dict) else str(err)
                else:
                    status_text = raw_body.strip()
            except Exception:
                status_text = None
            return code, str(status_text) if status_text is not None else None
        except (urllib.error.URLError, OSError, TimeoutError):
            return None, None

    def wait_until_ready(self, timeout: float = 120.0, poll_interval: float = 0.5) -> bool:
        """서버가 모델 로딩을 완료하고 HTTP 200 OK 상태가 될 때까지 폴링합니다.

        Args:
            timeout: 최대 대기 시간(초). 기본 120초.
            poll_interval: 폴링 주기(초). 기본 0.5초.

        Returns:
            bool: 정상 준비 완료 시 True.

        Raises:
            RuntimeError: 프로세스 비정상 종료(크래시) 또는 서버 내부 오류(HTTP 500) 발생 시.
            TimeoutError: 지정된 시간 내에 준비 완료되지 않은 경우.
        """
        start_time = time.monotonic()
        while True:
            # 1. Fail-fast: 관리 중인 자식 프로세스가 비정상 종료된 경우 즉시 예외 발생
            if self._proc is not None and self._proc.poll() is not None:
                exit_code = self._proc.poll()
                log_path = self.tools_dir / "llama-server.log"
                detail = ""
                if log_path.is_file():
                    try:
                        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        err_lines = [
                            l.strip()
                            for l in lines[-20:]
                            if any(k in l for k in ("Error", "error", "failed", "abort", "exception", "Exception"))
                        ]
                        if err_lines:
                            detail = " (원인: " + "; ".join(err_lines[-2:]) + ")"
                    except Exception:
                        pass
                raise RuntimeError(
                    f"llama-server 프로세스가 비정상 종료되었습니다 (exit code: {exit_code}).{detail} 로그 파일: {log_path}"
                )

            # 2. /health 엔드포인트 헬스체크 조회
            code, status = self._get_health_status()

            if code == 200 and status and status.lower() == "ok":
                print(f"[llama] 서버 준비 완료 (200 OK): {self.base_url}", flush=True)
                return True

            if code == 503:
                elapsed = time.monotonic() - start_time
                print(
                    f"[llama] 모델 로딩 대기 중... ({status or 'loading model'}, 경과: {elapsed:.1f}s / 제한: {timeout}s)",
                    flush=True,
                )
            elif code == 500 or (status and "error" in status.lower()):
                log_path = self.tools_dir / "llama-server.log"
                raise RuntimeError(
                    f"llama-server 내부 오류 발생 (HTTP {code}, 상태: {status}). 로그 파일: {log_path}"
                )

            # 3. 타임아웃 검사
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"llama-server 준비 대기 시간 초과 ({timeout}초 경과, 마지막 상태: HTTP {code}, {status})"
                )

            time.sleep(poll_interval)

    def _get_server_exe(self) -> Path:
        """tools_dir(및 레거시 경로)에서 llama-server 경로 탐색."""
        exe = _server_exe_name()
        for root in (self.tools_dir, _LEGACY_TOOLS_DIR, _SERVING_DIR):
            if not root.is_dir():
                continue
            direct = root / exe
            if direct.is_file():
                return direct
            for cand in root.rglob(exe):
                if cand.is_file():
                    return cand
        return self.tools_dir / exe

    def _download_file(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"[llama] 다운로드: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
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

    def _find_server_in(self, root: Path) -> Optional[Path]:
        """지정 디렉터리에서 llama-server 실행 파일 찾기."""
        if not root.is_dir():
            return None
        exe = _server_exe_name()
        direct = root / exe
        if direct.is_file():
            return direct
        for cand in root.rglob(exe):
            if cand.is_file():
                return cand
        return None

    def ensure_binary(self) -> Path:
        """llama-server 확인 → 레거시 복사 → 없으면 GitHub 다운로드."""
        found = self._find_server_in(self.tools_dir)
        if found is not None:
            return found

        # 포터블 경로로 이전 (tools/llama-cpp → data/serving_engines/llama-cpp)
        legacy = self._find_server_in(_LEGACY_TOOLS_DIR)
        if legacy is not None and self.tools_dir.resolve() != _LEGACY_TOOLS_DIR.resolve():
            print(f"[llama] 레거시 복사: {_LEGACY_TOOLS_DIR} → {self.tools_dir}")
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(_LEGACY_TOOLS_DIR, self.tools_dir, dirs_exist_ok=True)
            found = self._find_server_in(self.tools_dir)
            if found is not None:
                return found

        print(f"[llama] {self.tools_dir} 에 llama-server 가 없습니다. 최신 릴리스 다운로드 시작...")
        # GitHub Releases API로 최신 자산 조회
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"GitHub Releases 조회 실패 ({e})") from e

        is_win = sys.platform == "win32"
        variant = LLAMA_CPP_VARIANT.lower()
        if variant not in ("cpu", "vulkan", "cuda"):
            variant = "vulkan"

        # 타겟 자산 키워드 매칭
        chosen_asset = None
        for asset in data.get("assets", []):
            name = asset.get("name", "").lower()
            if is_win:
                if "bin-win" in name and variant in name and name.endswith(".zip"):
                    chosen_asset = asset
                    break
            else:
                if "bin-ubuntu" in name and variant in name and (name.endswith(".tar.gz") or name.endswith(".zip")):
                    chosen_asset = asset
                    break

        # fallback: generic asset
        if not chosen_asset:
            for asset in data.get("assets", []):
                name = asset.get("name", "").lower()
                if (is_win and "bin-win" in name and name.endswith(".zip")) or (not is_win and "bin" in name):
                    chosen_asset = asset
                    break

        if not chosen_asset:
            raise RuntimeError(f"적합한 llama.cpp 바이너리 자산을 찾을 수 없습니다 (variant={variant})")

        dl_url = chosen_asset["browser_download_url"]
        archive_name = chosen_asset["name"]
        archive_path = self.tools_dir / archive_name

        self._download_file(dl_url, archive_path)

        # 압축 해제
        print(f"[llama] 압축 해제 중: {archive_name}")
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(self.tools_dir)
        elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(self.tools_dir)

        if not is_win:
            server_exe = self._get_server_exe()
            if server_exe.is_file():
                server_exe.chmod(0o755)

        return self._get_server_exe()

    def ensure_model(self) -> Path:
        """data/models/chat 디렉터리에 GGUF 모델 가중치 확인 및 준비 (재다운로드 방지)."""
        CHAT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        gguf_path = CHAT_MODELS_DIR / LLAMA_GGUF_FILE
        min_bytes = 100 * 1024 * 1024  # 100MB 이상

        # 1. 표준 위치 확인
        if gguf_path.is_file() and gguf_path.stat().st_size > min_bytes:
            return gguf_path

        # 2. 레거시 위치 (data/models/{filename}) 확인 및 자동 이전
        legacy_path = MODELS_DIR / LLAMA_GGUF_FILE
        if legacy_path.is_file() and legacy_path.stat().st_size > min_bytes:
            print(f"[llama] 기존 다운로드 모델 감지: {legacy_path} -> {gguf_path} 이동")
            try:
                legacy_path.replace(gguf_path)
                return gguf_path
            except OSError:
                return legacy_path

        # 3. MODELS_DIR 하위 재귀 탐색 (기존 캐시나 다른 폴더에 이미 받아둔 경우)
        for cand in MODELS_DIR.rglob(LLAMA_GGUF_FILE):
            if cand.is_file() and cand.stat().st_size > min_bytes:
                print(f"[llama] 기존 다운로드 모델 감지: {cand} -> {gguf_path} 이동")
                try:
                    cand.replace(gguf_path)
                    return gguf_path
                except OSError:
                    return cand

        # 4. 없을 때만 Hugging Face 다운로드
        print(f"[llama] GGUF 모델 다운로드 시작: {LLAMA_GGUF_REPO} / {LLAMA_GGUF_FILE}")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=LLAMA_GGUF_REPO,
            filename=LLAMA_GGUF_FILE,
            local_dir=str(CHAT_MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        return gguf_path

    def _resolve_ngl(self, model_path: Path) -> int:
        """모델 크기 및 가용 GPU VRAM을 비교하여 안전한 ngl(GPU 오프로딩 레이어 수) 결정."""
        if LLAMA_NGL == 0:
            return 0
        try:
            from services.memory_manager import get_memory_facade
            facade = get_memory_facade()
            return facade.calculate_partial_ngl(model_path, total_layers=36)
        except Exception:
            try:
                from services.gpu_manager import calculate_safe_ngl, get_accelerator_manager
                acc_mgr = get_accelerator_manager()
                vram_bytes = acc_mgr.get_vram_bytes()
                model_size = model_path.stat().st_size if model_path.is_file() else 0
                return calculate_safe_ngl(vram_bytes, model_size, total_layers=36, requested_ngl=LLAMA_NGL)
            except Exception:
                pass
        return LLAMA_NGL

    def _launch_process(self, exe: Path, model_path: Path, ngl: int, host: str, port: int) -> None:
        """llama-server 프로세스를 백그라운드로 실행."""
        mmap_flag = "--mmap" if LLAMA_MMAP else "--no-mmap"
        cmd = [
            str(exe),
            "-m", str(model_path),
            "-c", str(LLAMA_CTX_SIZE),
            "-ngl", str(ngl),
            mmap_flag,
            "--host", host,
            "--port", str(port),
        ]

        print(f"[llama] 서버 기동: {' '.join(cmd)}")
        log_path = self.tools_dir / "llama-server.log"
        log_f = log_path.open("a", encoding="utf-8", errors="replace")
        popen_kw: dict = {"stdout": log_f, "stderr": subprocess.STDOUT}
        if sys.platform == "win32":
            detached = 0x00000008  # DETACHED_PROCESS
            new_group = 0x00000200  # CREATE_NEW_PROCESS_GROUP
            breakaway = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB
            popen_kw["close_fds"] = True
            try:
                self._proc = subprocess.Popen(cmd, creationflags=detached | new_group | breakaway, **popen_kw)
            except OSError:
                self._proc = subprocess.Popen(cmd, creationflags=detached | new_group, **popen_kw)
        else:
            self._proc = subprocess.Popen(cmd, start_new_session=True, **popen_kw)
        log_f.close()

    def start(self, timeout: float = 120.0) -> bool:
        """바이너리와 모델을 준비하고 llama-server를 실행하며 준비 완료(200 OK)를 대기합니다."""
        if self.is_running():
            print("[llama] llama-server 가 이미 실행 중입니다. 준비 상태를 확인합니다.")
            return self.wait_until_ready(timeout=timeout)

        exe = self.ensure_binary()
        model_path = self.ensure_model()

        parsed = urlparse(self.base_url)
        port = parsed.port or 8080
        host = parsed.hostname or "127.0.0.1"

        ngl = self._resolve_ngl(model_path)
        self._launch_process(exe, model_path, ngl, host, port)

        try:
            return self.wait_until_ready(timeout=timeout)
        except RuntimeError as exc:
            # GPU OOM 또는 모델 로딩 실패로 비정상 종료 시 ngl > 0 이었으면 CPU(-ngl 0)로 자동 치유(Auto-healing) 폴백
            if ngl > 0:
                print(
                    f"[llama] [경고] GPU 가속 기동 실패 ({exc}). "
                    f"GPU 메모리 부족으로 판단하여 CPU 모드(-ngl 0)로 자동 폴백하여 재시도합니다...",
                    flush=True,
                )
                self._launch_process(exe, model_path, 0, host, port)
                return self.wait_until_ready(timeout=timeout)
            raise
