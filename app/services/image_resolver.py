"""RAG 검색 결과(Hits)에 연관된 문서 이미지 탐색 및 해석기."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from settings import PARSED_DIR


class ImageResolver:
    """문서 파싱 JSON에서 청크 페이지에 해당하는 이미지 및 캡션을 추출/매핑하는 클래스."""

    def __init__(self, parsed_dir: Path = PARSED_DIR) -> None:
        self.parsed_dir = parsed_dir
        self._doc_image_cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    def _load_doc_images(self, doc_stem: str) -> Dict[int, List[Dict[str, Any]]]:
        """문서 JSON을 읽어 {page_number: [{'source': str, 'caption': str, 'bbox': list}]} 매핑 반환."""
        if doc_stem in self._doc_image_cache:
            return self._doc_image_cache[doc_stem]

        json_path = self.parsed_dir / f"{doc_stem}.json"
        if not json_path.is_file():
            candidates = list(self.parsed_dir.glob(f"{doc_stem}*.json"))
            if candidates:
                json_path = candidates[0]
            else:
                self._doc_image_cache[doc_stem] = {}
                return {}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._doc_image_cache[doc_stem] = {}
            return {}

        page_images: Dict[int, List[Dict[str, Any]]] = {}
        images_by_id: Dict[int, Dict[str, Any]] = {}

        def walk(node: Any, current_page: Optional[int] = None) -> None:
            if isinstance(node, dict):
                p = node.get("page number") or node.get("page_number") or current_page
                node_type = node.get("type")
                node_id = node.get("id")

                if node_type == "image" and node.get("source"):
                    img_data = {
                        "source": node["source"],
                        "page": p,
                        "bbox": node.get("bounding box") or node.get("bbox"),
                        "caption": "",
                    }
                    if p not in page_images:
                        page_images[p] = []
                    page_images[p].append(img_data)
                    if node_id is not None:
                        images_by_id[node_id] = img_data

                elif node_type == "caption":
                    linked_id = node.get("linked content id")
                    cap_text = (node.get("content") or "").strip()
                    if linked_id in images_by_id and cap_text:
                        images_by_id[linked_id]["caption"] = cap_text

                for v in node.values():
                    walk(v, p)
            elif isinstance(node, list):
                for item in node:
                    walk(item, current_page)

        walk(data)
        self._doc_image_cache[doc_stem] = page_images
        return page_images

    def resolve_images_for_hits(
        self,
        hits: List[Dict[str, Any]],
        max_images: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        리랭크된 청크 리스트로부터 관련된 이미지 목록을 추출합니다.
        """
        from settings import SHOW_DOC_IMAGES, MAX_DOC_IMAGES

        if not SHOW_DOC_IMAGES or not hits:
            return []

        limit = max_images if max_images is not None else MAX_DOC_IMAGES
        matched_images: List[Dict[str, Any]] = []
        seen_sources: Set[str] = set()

        for hit in hits:
            file_name = hit.get("file_name") or ""
            pages = hit.get("pages") or ([hit.get("page")] if hit.get("page") is not None else [])
            if not file_name or not pages:
                continue

            stem = Path(file_name).stem
            doc_map = self._load_doc_images(stem)

            imgs_on_page: List[Dict[str, Any]] = []
            for p in pages:
                imgs_on_page.extend(doc_map.get(p, []))

            for img in imgs_on_page:
                src = img.get("source")
                if not src or src in seen_sources:
                    continue

                full_path = self.parsed_dir / src
                if not full_path.is_file():
                    continue

                seen_sources.add(src)
                matched_images.append({
                    "url": f"/api/images/{src}",
                    "file_name": file_name,
                    "page": img.get("page") or hit.get("page"),
                    "caption": img.get("caption", ""),
                    "source": src,
                })

                if len(matched_images) >= limit:
                    return matched_images

        return matched_images


_resolver: Optional[ImageResolver] = None


def get_image_resolver() -> ImageResolver:
    """ImageResolver 싱글톤 인스턴스 반환."""
    global _resolver
    if _resolver is None:
        _resolver = ImageResolver()
    return _resolver

