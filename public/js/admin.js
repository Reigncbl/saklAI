// js/admin.js

/********************
 * DATA
 ********************/
const autoChatCards = [
    { user: "User #1", inquiry: "IDK BRO", percent: 50, time: "2 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #2", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "assigned", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #3", inquiry: "Loan Info Inquiry", percent: 0, time: "2 mins ago", status: "complex", customerMsg: "Customer", agentMsg: "" },
    { user: "User #4", inquiry: "Loan Info Inquiry", percent: 75, time: "3 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #5", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #6", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" }
];

const activeChatCards = [
    { user: "User #20", inquiry: "Loan Application", percent: 100, time: "just now", status: "assigned", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #82", inquiry: "Account Closure", percent: 20, time: "1 min ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
    { user: "User #90", inquiry: "Loan Application", percent: 0, time: "2 mins ago", status: "complex", customerMsg: "Customer", agentMsg: "" },
    { user: "User #45", inquiry: "Loan Info Inquiry", percent: 75, time: "3 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" }
];

const sidebarChats = [
    { user: "User #78", subtitle: "Loan Application", active: true },
    { user: "User #82", subtitle: "Account Closure", active: false },
    { user: "User #90", subtitle: "Loan Application", active: false }
];

const chatTemplates = {
    "User #78": {
        name: "Tessa Erika de Guzman",
        messages: [
            { label: "Customer", bubble: "Hi, I need help with my loan application.", agent: false },
            { label: "", bubble: "Complex Case Detected. Assigning to human agent...", yellow: true },
            { label: "Agent John Doe", bubble: "Hello Tessa, I'll help you with this case.", agent: true }
        ]
    },
    "User #82": {
        name: "Juan Dela Cruz",
        messages: [
            { label: "Customer", bubble: "I want to close my account.", agent: false },
            { label: "Agent John Doe", bubble: "I can assist you with the process.", agent: true }
        ]
    },
    "User #90": {
        name: "Maria Santos",
        messages: [
            { label: "Customer", bubble: "What's the status of my loan?", agent: false },
            { label: "Agent John Doe", bubble: "Let me check your application details.", agent: true }
        ]
    }
};

const recommendations = [
    { title: "CLARIFY", text: "To help you get the best loan terms, I'll need to review your financial statements and bank records. Do you have 2023–2024 income records available?" },
    { title: "PRODUCT RECOMMENDATION", text: "Based on your business type, I'd recommend our SME Business Growth Loan with flexible payment terms." },
    { title: "NEXT STEPS", text: "Would you like me to schedule a detailed consultation call or prepare a pre-qualification assessment?" }
];

/********************
 * RENDER HELPERS
 ********************/
const API_BASE_URL = (() => {
  if (window.location.protocol === 'file:') return 'http://localhost:8000';
  if (window.location.hostname === 'localhost' && window.location.port !== '8000') return 'http://localhost:8000';
  return '';
})();

const el = (id) => document.getElementById(id);

// Auto Chats - Reusable Component
function renderCard(card) {
    const isYellow = ["assigned", "complex"].includes(card.status);
    const percentClass =
        card.percent === 0 ? "percent zero" :
            isYellow ? "percent yellow" :
                card.percent < 100 ? "percent low" : "percent";

  // Render all chat bubbles from history if available
  let chatBubbles = "";
  if (Array.isArray(card.history)) {
    chatBubbles = card.history.map(m => {
      let bubbles = [];
      let bubbleClass = "chat-bubble";
      if (m.role === "assistant") bubbleClass += " agent";
      if (m.role === "system") bubbleClass += " system";
      // Main message content
      if (m.content) {
        bubbles.push(`<div class="${bubbleClass}">${m.content}</div>`);
      }
      // If assistant and has suggestions, render each as a bubble
      if (m.role === "assistant" && m.response && Array.isArray(m.response.suggestions)) {
        m.response.suggestions.forEach(sug => {
          if (sug.suggestion) {
            bubbles.push(`<div class="chat-bubble agent suggestion">${sug.suggestion}</div>`);
          }
        });
      }
      return bubbles.join("");
    }).join("");
  } else {
    chatBubbles = `<div class="chat-bubble">${card.customerMsg}</div>` +
      (card.status === "complex"
        ? `<div class="chat-bubble yellow">Complex Case Detected. Assigning to human agent...</div>`
        : `<div class="chat-bubble agent">${card.agentMsg}</div>`);
  }

    return `
    <div class="chat-card${isYellow ? " yellow" : ""}">
      <div>
        <div class="chat-header">
          <div>
            <div class="chat-user">${card.user}</div>
            <div class="chat-type">${card.inquiry}</div>
          </div>
          <div class="chat-meta">
            <span class="time">${card.time}</span>
          </div>
        </div>
        <div class="chat-content">
          ${chatBubbles}
        </div>
      </div>
      <div class="chat-footer">
        ${card.status === "takeover"
            ? `<button class="btn takeover" data-user="${card.user}">Take Over</button>`
            : `<button class="btn assigned" disabled>Assigned</button>`}
      </div>
    </div>`;
}

// Layout
function renderGrid(cards) {
    el("chatGrid").innerHTML = cards.map(renderCard).join("");
}

// For panel 1 content
async function fetchActiveChats() {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/active`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const chats = await response.json();
    console.log("Chats API response:", chats);

    // Map backend data to card format expected by renderGrid
    const cards = chats.map(c => ({
      user: c.user_id,
      inquiry: "Customer Inquiry", // Placeholder, backend does not provide
      time: c.last_timestamp ? new Date(c.last_timestamp).toLocaleTimeString() : "",
      status: "takeover",
      history: c.history // Pass full chat history
    }));

    renderGrid(cards);
  } catch (err) {
    console.error("Failed to fetch active chats:", err);
    // Optionally show a message in the grid
    el("chatGrid").innerHTML = `<div class='error'>Failed to load active chats.</div>`;
  }
}


// PANEL 2

function renderSidebar(chats) {
    return `
    <div class="sidebar-header">
      <span>Active Human Chats</span><span>${chats.length}</span>
    </div>
    <ul class="sidebar-list">
      ${chats.map((c, i) => `
        <li data-index="${i}" class="${c.active ? "active" : ""}">
          ${c.user}<div class="subtitle">${c.subtitle}</div>
        </li>`).join("")}
    </ul>`;
}

function renderChatPanel(chat) {
    return `
    <div class="chat-header">${chat.name}</div>
    <div class="chat-messages">
      ${chat.messages.map(m => `
        <div class="chat-row${m.agent ? " agent" : ""}">
          ${m.label ? `<div class="chat-label${m.agent ? " agent" : ""}">${m.label}</div>` : ""}
          ${m.bubble ? `<div class="chat-bubble${m.agent ? " agent" : ""}${m.yellow ? " yellow" : ""}">${m.bubble}</div>` : ""}
        </div>`).join("")}
    </div>
    <form class="chat-input-area" onsubmit="event.preventDefault();">
      <input type="text" class="chat-input" placeholder="Type your message here..." autocomplete="off" />
      <button type="submit" class="chat-send">Send</button>
    </form>`;
}

function renderRecPanel(list) {
    return `
    <div class="rec-header">AI Message Recommendations</div>
    <div class="rec-list">
      ${list.map(r => `
        <div class="rec-item">
          <div class="rec-title">${r.title}</div>
          <div class="rec-text">${r.text}</div>
          <button class="rec-use-btn" type="button">Use</button>
        </div>`).join("")}
    </div>`;
}

function mountPanel2() {
    el("sidebar").innerHTML = renderSidebar(sidebarChats);
    const activeUser = sidebarChats.find(c => c.active)?.user || sidebarChats[0].user;
    el("chatPanel").innerHTML = renderChatPanel(chatTemplates[activeUser]);
    el("recPanel").innerHTML = renderRecPanel(recommendations);
}

/********************
 * NAVIGATION / EVENTS
 ********************/
function switchTab(on, off, show, hide, fn) {
    el(on).classList.add("active");
    el(off).classList.remove("active");
    el(show).classList.remove("hidden");
    el(hide).classList.add("hidden");
    fn?.();
}

// Only use fetchActiveChats for rendering cards on load and on tab switch
window.addEventListener("load", () => {
  fetchActiveChats();
  setInterval(fetchActiveChats, 5000); // Matic reload
});

function setTabActive(tabId, otherTabId) {
  el(tabId).classList.add("active");
  el(otherTabId).classList.remove("active");
}

function showPanel(panelNum) {
  const panel1 = el("panel1");
  const panel2 = el("panel2");
  if (panelNum === 1) {
    panel1.classList.remove("hidden");
    panel2.classList.add("hidden");
    setTabActive("autoTab", "activeTab");
  } else {
    panel1.classList.add("hidden");
    panel2.classList.remove("hidden");
    setTabActive("activeTab", "autoTab");
  }
}

el("autoTab").onclick = () => { showPanel(1); fetchActiveChats(); };
el("activeTab").onclick = () => { showPanel(2); mountPanel2(); };

el("autoTab2").onclick = () => { showPanel(1); fetchActiveChats(); };
el("activeTab2").onclick = () => { showPanel(2); mountPanel2(); };

document.addEventListener("click", (e) => {

  if (e.target.closest(".btn.takeover")) {
    const btn = e.target.closest(".btn.takeover");
    const userId = btn.getAttribute("data-user");
    // Call backend to set status to 'assigned'
    fetch(`${API_BASE_URL}/chat/status/${userId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "assigned" })
    })
    .then(res => {
      if (!res.ok) throw new Error("Failed to assign chat");
      // Refresh panel 1 (dashboard)
      fetchActiveChats();
      // Optionally, switch to panel 2 after update
      switchTab("activeTab", "autoTab", "panel2", "panel1", mountPanel2);
    })
    .catch(err => {
      alert("Failed to assign chat: " + err.message);
    });
  }

    const li = e.target.closest("#sidebar .sidebar-list li");
    if (li) {
        sidebarChats.forEach((c, i) => c.active = i === +li.dataset.index);
        el("sidebar").innerHTML = fetchActiveChats();
        const user = sidebarChats.find(c => c.active).user;
        el("chatPanel").innerHTML = renderChatPanel(chatTemplates[user]);
    }
});
