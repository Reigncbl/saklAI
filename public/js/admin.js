// js/admin.js

/********************
 * DATA
 ********************/
const autoChatCards = [
    { user: "User #1", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
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
const el = (id) => document.getElementById(id);

function renderCard(card) {
    const isYellow = ["assigned", "complex"].includes(card.status);
    const percentClass =
        card.percent === 0 ? "percent zero" :
            isYellow ? "percent yellow" :
                card.percent < 100 ? "percent low" : "percent";

    return `
    <div class="chat-card${isYellow ? " yellow" : ""}">
      <div>
        <div class="chat-header">
          <div>
            <div class="chat-user">${card.user}</div>
            <div class="chat-type">${card.inquiry}</div>
          </div>
          <div class="chat-meta">
            <span class="${percentClass}">${card.percent}%</span>
            <span class="time">${card.time}</span>
          </div>
        </div>
        <div class="chat-content">
          <div class="chat-bubble">${card.customerMsg}</div>
          ${card.status === "complex"
            ? `<div class="chat-bubble yellow">Complex Case Detected. Assigning to human agent...</div>`
            : `<div class="chat-bubble agent">${card.agentMsg}</div>`}
        </div>
      </div>
      <div class="chat-footer">
        ${card.status === "takeover"
            ? `<button class="btn takeover" data-user="${card.user}">Take Over</button>`
            : `<button class="btn assigned" disabled>Assigned</button>`}
      </div>
    </div>`;
}

function renderGrid(cards) {
    el("chatGrid").innerHTML = cards.map(renderCard).join("");
}

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

renderGrid(autoChatCards);

el("autoTab").onclick = () => switchTab("autoTab", "activeTab", "panel1", "panel2", () => renderGrid(autoChatCards));
el("activeTab").onclick = () => switchTab("activeTab", "autoTab", "panel2", "panel1", mountPanel2);

el("autoTab2").onclick = () => switchTab("autoTab2", "activeTab2", "panel1", "panel2", () => renderGrid(autoChatCards));
el("activeTab2").onclick = () => switchTab("activeTab2", "autoTab2", "panel2", "panel1", mountPanel2);

document.addEventListener("click", (e) => {
    if (e.target.closest(".btn.takeover")) switchTab("activeTab", "autoTab", "panel2", "panel1", mountPanel2);

    const li = e.target.closest("#sidebar .sidebar-list li");
    if (li) {
        sidebarChats.forEach((c, i) => c.active = i === +li.dataset.index);
        el("sidebar").innerHTML = renderSidebar(sidebarChats);
        const user = sidebarChats.find(c => c.active).user;
        el("chatPanel").innerHTML = renderChatPanel(chatTemplates[user]);
    }
});
