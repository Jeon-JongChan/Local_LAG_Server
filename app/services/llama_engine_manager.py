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
from typing import Optional
from urllib.parse import urlparse

from .base import BaseEngineManager

try:
    from settings import (
        BASE_DIR,
        CHAT_MODELS_DIR,
        LLAMA_CPP_BASE_URL,
        LLAMA_CPP_VARIANT,
        LLAMA_CTX_SIZE,
        LLAMA_GGUF_FILE,
        LLAMA_GGUF_REPO,
        LLAMA_NGL,
        MODELS_DIR,
    )
except ModuleNotFoundError:
    from app.settings import (
        BASE_DIR,
        CHAT_MODELS_DIR,
        LLAMA_CPP_BASE_URL,
        LLAMA_CPP_VARIANT,
        LLAMA_CTX_SIZE,
        LLAMA_GGUF_FILE,
        LLAMA_GGUF_REPO,
        LLAMA_NGL,
        MODELS_DIR,
    )

_TOOLS_DIR = BASE_DIR / "tools" / "llama-cpp"
_USER_AGENT = "local-rag/2.0 (llama-engine-manager)"


class LlamaEngineManager(BaseEngineManager):
    """llama.cpp 바이너리 및 런타임 관리자."""

    def __init__(self, base_url: str = LLAMA_CPP_BASE_URL, tools_dir: Path = _TOOLS_DIR) -> None:
        self.base_url = base_url.rstrip("/")
        self.tools_dir = tools_dir
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def is_running(self) -> bool:
        """포트 열림 및 헬스체크 확인."""
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

    def _get_server_exe(self) -> Path:
        exe = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        direct = self.tools_dir / exe
        if direct.is_file():
            return direct
        for cand in self.tools_dir.rglob(exe):
            if cand.is_file():
                return cand
        return direct

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

    def ensure_binary(self) -> Path:
        """llama-server 바이너리 존재 여부 확인 및 GitHub 자동 다운로드."""
        server_exe = self._get_server_exe()
        if server_exe.is_file():
            return server_exe

        print(f"[llama] tools/llama-cpp 에 llama-server 가 없습니다. 최신 릴리스 다운로드 시작...")
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

    def start(self) -> bool:
        """바이너리와 모델을 준비하고 llama-server를 실행합니다."""
        if self.is_running():
            print("[llama] llama-server 가 이미 실행 중입니다.")
            return True

        exe = self.ensure_binary()
        model_path = self.ensure_model()

        parsed = urlparse(self.base_url)
        port = parsed.port or 8080
        host = parsed.hostname or "127.0.0.1"

        cmd = [
            str(exe),
            "-m", str(model_path),
            "-c", str(LLAMA_CTX_SIZE),
            "-ngl", str(LLAMA_NGL),
            "--host", host,
            "--port", str(port),
        ]

        print(f"[llama] 서버 기동: {' '.join(cmd)}")
        if sys.platform == "win32":
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 헬스체크 대기 (최대 30초)
        for _ in range(30):
            time.sleep(1.0)
            if self.is_running():
                print(f"[llama] 서버 기동 성공 ({self.base_url})")
                return True

        return self.is_running()
