/**
 * local-rag — Vue 3 ChatGPT 스타일 채팅 UI
 */

const { createApp, ref, computed, nextTick, onMounted } = Vue;

const REF_MARKER = "\n\n[참고 문서]\n";

function stripMarkdownBold(text) {
  if (!text || !text.includes("**")) return text;
  return text
    .split(/(```[\s\S]*?```|`[^`\n]+`)/)
    .map((part, i) => (i % 2 === 1 ? part : part.replaceAll("**", "")))
    .join("");
}

function parseAssistantContent(text) {
  const idx = text.indexOf(REF_MARKER);
  let rawBody = idx === -1 ? text : text.slice(0, idx).trim();
  const body = stripMarkdownBold(rawBody);
  const refs = idx === -1 ? [] : text
    .slice(idx + REF_MARKER.length)
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- "))
    .map((line) => line.slice(2));
  return { body, refs };
}

createApp({
  setup() {
    const sessions = ref([]);
    const currentSessionId = ref(null);
    const messages = ref([]);
    const input = ref("");
    const thinkMode = ref("instant");
    const useHistory = ref(false);
    const reason = ref(false);
    const loading = ref(false);
    const liveLines = ref([]);
    let liveTimer = null;
    let sessionCheckTimer = null;
    const sidebarOpen = ref(window.innerWidth > 768);
    const messagesEl = ref(null);
    const inputEl = ref(null);
    const modalImageUrl = ref(null);
    const modalImageCaption = ref("");

    function openImageModal(url, caption = "") {
      modalImageUrl.value = url;
      modalImageCaption.value = caption;
    }

    function closeImageModal() {
      modalImageUrl.value = null;
      modalImageCaption.value = "";
    }

    const hints = [
      "DCU 구매규격 요약해줘",
      "원격검침 자동전환 조건은?",
      "AMI 운영자 메뉴얼 핵심만",
    ];

    const currentTitle = computed(() => {
      const s = sessions.value.find((x) => x.id === currentSessionId.value);
      return s?.title || "새 대화";
    });

    const loadingLabel = computed(() => {
      if (!reason.value) return "임베딩 · 리랭크 · 청크 선별…";
      if (thinkMode.value === "think") {
        return "임베딩 · 리랭크 · Thinking…";
      }
      if (thinkMode.value === "auto") {
        return "임베딩 · 리랭크 · Auto 생성…";
      }
      return "임베딩 · 리랭크 · 생성 중…";
    });

    function parsed(msg) {
      if (msg.role !== "assistant") {
        return { body: msg.content, refs: [] };
      }
      return parseAssistantContent(msg.content);
    }

    async function api(method, path, body) {
      const opts = { method, headers: { "Content-Type": "application/json" } };
      if (body !== undefined) {
        opts.body = JSON.stringify(body);
      }
      const res = await fetch(path, opts);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      return data;
    }

    async function loadSessions() {
      const data = await api("GET", "/api/sessions");
      sessions.value = data.sessions || [];
    }

    function stopSessionPoll() {
      clearInterval(sessionCheckTimer);
      sessionCheckTimer = null;
    }

    function startSessionPoll(sid) {
      stopSessionPoll();
      sessionCheckTimer = setInterval(async () => {
        if (currentSessionId.value !== sid) {
          stopSessionPoll();
          return;
        }
        try {
          const data = await api("GET", `/api/sessions/${sid}`);
          if (currentSessionId.value !== sid) return;
          if (!data.generating) {
            stopSessionPoll();
            stopLivePoll();
            loading.value = false;
            messages.value = data.messages || [];
            await scrollBottom();
          }
        } catch {
          /* 주기적 재시도 */
        }
      }, 1500);
    }

    async function selectSession(id) {
      stopSessionPoll();
      currentSessionId.value = id;
      const data = await api("GET", `/api/sessions/${id}`);
      messages.value = data.messages || [];
      if (window.innerWidth <= 768) sidebarOpen.value = false;
      await scrollBottom();

      if (data.generating) {
        loading.value = true;
        startLivePoll();
        startSessionPoll(id);
      } else {
        loading.value = false;
        stopLivePoll();
      }
    }

    async function newChat() {
      const data = await api("POST", "/api/sessions", {});
      sessions.value.unshift(data.session);
      currentSessionId.value = data.session.id;
      messages.value = [];
      input.value = "";
      if (window.innerWidth <= 768) sidebarOpen.value = false;
      await nextTick(() => inputEl.value?.focus());
    }

    async function deleteSession(id) {
      if (!confirm("이 대화를 삭제할까요?")) return;
      await api("DELETE", `/api/sessions/${id}`);
      sessions.value = sessions.value.filter((s) => s.id !== id);
      if (currentSessionId.value === id) {
        currentSessionId.value = null;
        messages.value = [];
        if (sessions.value.length) {
          await selectSession(sessions.value[0].id);
        }
      }
    }

    async function scrollBottom() {
      await nextTick();
      const el = messagesEl.value;
      if (el) el.scrollTop = el.scrollHeight;
    }

    function resizeInput() {
      const el = inputEl.value;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 200) + "px";
    }

    function onKeydown(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    }

    function useHint(text) {
      input.value = text;
      resizeInput();
      inputEl.value?.focus();
    }

    function startLivePoll() {
      liveLines.value = [];
      clearInterval(liveTimer);
      liveTimer = setInterval(async () => {
        try {
          const data = await api("GET", "/api/query/live");
          liveLines.value = data.lines || [];
          await scrollBottom();
        } catch {
          /* 폴링 실패는 다음 주기에 재시도 */
        }
      }, 1500);
    }

    function stopLivePoll() {
      clearInterval(liveTimer);
      liveTimer = null;
      liveLines.value = [];
    }

    async function send() {
      const text = input.value.trim();
      if (!text || loading.value) return;

      loading.value = true;
      startLivePoll();
      input.value = "";
      resizeInput();

      try {
        if (!currentSessionId.value) {
          const created = await api("POST", "/api/sessions", {});
          sessions.value.unshift(created.session);
          currentSessionId.value = created.session.id;
        }

        const sid = currentSessionId.value;
        messages.value.push({
          id: "tmp-u-" + Date.now(),
          role: "user",
          content: text,
        });
        await scrollBottom();

        const data = await api("POST", `/api/sessions/${sid}/messages`, {
          content: text,
          think_mode: thinkMode.value,
          use_history: useHistory.value,
          reason: reason.value,
        });

        // 세션 목록 순서 및 제목 갱신
        if (data.session) {
          const idx = sessions.value.findIndex((s) => s.id === sid);
          if (idx >= 0) {
            sessions.value[idx] = data.session;
            const [updated] = sessions.value.splice(idx, 1);
            sessions.value.unshift(updated);
          }
        }

        // 사용자가 아직 해당 채팅방에 머물러 있는 경우에만 현재 화면 갱신
        if (currentSessionId.value === sid) {
          messages.value = messages.value.filter((m) => !String(m.id).startsWith("tmp-"));
          if (data.user_message && data.assistant_message) {
            messages.value.push(data.user_message, data.assistant_message);
          }
          await scrollBottom();
        }
      } catch (err) {
        // 사용자가 방을 이동하여 요청이 취소된 경우 에러 메시지를 띄우지 않고 백그라운드 완료에 맡김
        if (currentSessionId.value) {
          const data = await api("GET", `/api/sessions/${currentSessionId.value}`).catch(() => null);
          if (data && data.generating) {
            startSessionPoll(currentSessionId.value);
            return;
          }
        }
        messages.value.push({
          id: "err-" + Date.now(),
          role: "assistant",
          content: `오류: ${err.message}`,
        });
        await scrollBottom();
      } finally {
        if (!sessionCheckTimer) {
          stopLivePoll();
          loading.value = false;
        }
        await nextTick(() => inputEl.value?.focus());
      }
    }

    onMounted(async () => {
      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && modalImageUrl.value) {
          closeImageModal();
        }
      });
      try {
        await loadSessions();
        if (sessions.value.length) {
          await selectSession(sessions.value[0].id);
        }
      } catch (err) {
        console.error(err);
      }
      inputEl.value?.focus();
    });

    return {
      sessions,
      currentSessionId,
      messages,
      input,
      thinkMode,
      useHistory,
      reason,
      loading,
      loadingLabel,
      liveLines,
      sidebarOpen,
      messagesEl,
      inputEl,
      hints,
      currentTitle,
      parsed,
      newChat,
      selectSession,
      deleteSession,
      send,
      onKeydown,
      resizeInput,
      useHint,
      modalImageUrl,
      modalImageCaption,
      openImageModal,
      closeImageModal,
    };
  },
}).mount("#app");
