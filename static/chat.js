const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const toolsBadge = document.getElementById("tools-badge");

function addMessage(role, text, tags = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (tags.length > 0) {
    const tagsDiv = document.createElement("div");
    tagsDiv.className = "tags";
    tags.forEach((tag) => {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = tag;
      tagsDiv.appendChild(span);
    });
    bubble.appendChild(tagsDiv);
  }

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return wrapper;
}

function addLoadingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message agent";
  wrapper.id = "loading-message";

  const bubble = document.createElement("div");
  bubble.className = "bubble loading";
  bubble.textContent = "Escribiendo…";

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeLoadingMessage() {
  const el = document.getElementById("loading-message");
  if (el) el.remove();
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      const docs = data.documents ?? "?";
      const chunks = data.chunks ?? "?";
      toolsBadge.textContent = `${docs} documentos · ${chunks} fragmentos · ${data.chat_model}`;
    } else {
      toolsBadge.textContent = "agente no disponible";
    }
  } catch (err) {
    toolsBadge.textContent = "sin conexión";
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  chatInput.value = "";
  sendBtn.disabled = true;
  addLoadingMessage();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    removeLoadingMessage();

    if (data.error) {
      addMessage("agent", `⚠️ ${data.error}`);
    } else {
      addMessage("agent", data.answer, data.sources || []);
    }
  } catch (err) {
    removeLoadingMessage();
    addMessage("agent", "⚠️ Error de conexión con el servidor. Intenta de nuevo.");
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
});

loadHealth();
