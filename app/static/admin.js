/** 관리자 페이지: 설정 스냅샷 + 질의 구간 시간 피드. */

(function () {
  const cfgEl = document.getElementById("config");
  const feedEl = document.getElementById("feed");
  const emptyEl = document.getElementById("feed-empty");
  const countEl = document.getElementById("feed-count");

  const CFG_LABELS = {
    profile: "PROFILE",
    accelerator: "ACCELERATOR",
    serving_engine: "서빙 엔진",
    embed_model: "임베딩",
    embed_device: "임베딩 디바이스",
    rerank_strategy: "리랭크 전략",
    rerank_model: "리랭크 모델",
    doc_parser: "파서",
    faiss_top_k: "FAISS Top-K",
    rerank_top_k: "Rerank Top-K",
    hybrid_bm25: "Hybrid BM25",
  };

  /**
   * API 로드 후 화면 갱신.
   */
  async function load() {
    const res = await fetch("/api/admin");
    if (!res.ok) throw new Error("admin api " + res.status);
    const data = await res.json();
    renderConfig(data.config || {});
    renderFeed(data.traces || []);
  }

  /**
   * 상단 설정 카드.
   * @param {Record<string, unknown>} cfg
   */
  function renderConfig(cfg) {
    cfgEl.innerHTML = "";
    Object.keys(CFG_LABELS).forEach((key) => {
      const card = document.createElement("div");
      card.className = "cfg-card";
      const k = document.createElement("div");
      k.className = "k";
      k.textContent = CFG_LABELS[key];
      const v = document.createElement("div");
      v.className = "v";
      const val = cfg[key];
      v.textContent = val === undefined || val === null || val === "" ? "—" : String(val);
      card.appendChild(k);
      card.appendChild(v);
      cfgEl.appendChild(card);
    });
  }

  /**
   * 질의 피드 (접힌 카드).
   * @param {Array<Record<string, unknown>>} traces
   */
  function renderFeed(traces) {
    feedEl.innerHTML = "";
    countEl.textContent = traces.length ? `(${traces.length})` : "";
    emptyEl.hidden = traces.length > 0;

    traces.forEach((tr) => {
      const stages = Array.isArray(tr.stages) ? tr.stages : [];
      const details = document.createElement("details");
      details.className = "card";

      const summary = document.createElement("summary");
      const row1 = document.createElement("div");
      row1.className = "row1";
      const q = document.createElement("div");
      q.className = "q";
      q.textContent = tr.question || "(빈 질문)";
      const total = document.createElement("div");
      total.className = "total";
      total.textContent = `${Number(tr.total_s || 0).toFixed(2)}s`;
      row1.appendChild(q);
      row1.appendChild(total);

      const chips = document.createElement("div");
      chips.className = "chips";
      if (!tr.ok) {
        const bad = document.createElement("span");
        bad.className = "chip bad";
        bad.textContent = "실패";
        chips.appendChild(bad);
      }
      stages.forEach((st) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = `${st.label} ${Number(st.s || 0).toFixed(2)}s`;
        chips.appendChild(chip);
      });

      summary.appendChild(row1);
      summary.appendChild(chips);

      const detail = document.createElement("div");
      detail.className = "detail";
      stages.forEach((st) => {
        const row = document.createElement("div");
        row.className = "stage";
        row.innerHTML =
          `<div class="label">${escapeHtml(st.label || st.id || "")}</div>` +
          `<div class="model"><strong>${escapeHtml(st.model || "")}</strong>` +
          (st.detail ? `<div class="muted">${escapeHtml(st.detail)}</div>` : "") +
          `</div>` +
          `<div class="sec">${Number(st.s || 0).toFixed(2)}s</div>`;
        detail.appendChild(row);
      });

      const meta = document.createElement("div");
      meta.className = "meta";
      const eng = (tr.config && tr.config.serving_engine) || "";
      meta.textContent = `${tr.ts || ""} · engine=${eng} · think=${tr.think_mode || ""}`;
      detail.appendChild(meta);

      details.appendChild(summary);
      details.appendChild(detail);
      feedEl.appendChild(details);
    });
  }

  /**
   * @param {string} s
   * @returns {string}
   */
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  document.getElementById("btn-refresh").addEventListener("click", () => {
    load().catch((e) => {
      cfgEl.innerHTML = `<p class="muted">로드 실패: ${escapeHtml(e.message)}</p>`;
    });
  });

  load().catch((e) => {
    cfgEl.innerHTML = `<p class="muted">로드 실패: ${escapeHtml(e.message)}</p>`;
  });

  // 채팅 후 자동 갱신용 짧은 폴링
  setInterval(() => {
    load().catch(() => {});
  }, 8000);
})();
