const sidebarChats = [
    { user: "User #78", subtitle: "Loan Application", active: true },
    { user: "User #82", subtitle: "Account Closure", active: false },
    { user: "User #90", subtitle: "Loan Application", active: false }
];

function renderSidebar(chats) {
    return `
        <div class="sidebar-header">
          <span>Active Human Chats</span>
          <span>${chats.length}</span>
        </div>
        <ul class="sidebar-list">
          ${chats.map(chat => `
            <li class="${chat.active ? 'active' : ''}">
              ${chat.user}
              <div class="subtitle">${chat.subtitle}</div>
            </li>
          `).join('')}
        </ul>
      `;
}

// Chat panel data
const chatPanelData = {
    name: "Tessa Erika de Guzman",
    messages: [
        { label: "Customer", bubble: "", agent: false },
        { label: "", bubble: "Complex Case Detected. Assigning to human agent...", agent: false, yellow: true },
        { label: "Agent John Doe", bubble: "", agent: true },
        { label: "Customer", bubble: "", agent: false },
        { label: "Agent John Doe", bubble: "", agent: true }
    ]
};

function renderChatPanel(chat) {
    return `
        <div class="chat-header">
          ${chat.name}
        </div>
        <div class="chat-messages">
          ${chat.messages.map(msg => `
            <div class="chat-row${msg.agent ? ' agent' : ''}">
              ${msg.label ? `<div class="chat-label${msg.agent ? ' agent' : ''}">${msg.label}</div>` : ''}
              <div class="chat-bubble${msg.agent ? ' agent' : ''}${msg.yellow ? ' yellow' : ''}">${msg.bubble}</div>
            </div>
          `).join('')}
        </div>
        <form class="chat-input-area" onsubmit="event.preventDefault();">
          <input type="text" class="chat-input" placeholder="Type your message here..." autocomplete="off" />
          <button type="submit" class="chat-send">Send</button>
        </form>
      `;
}

// Recommendation panel data
const recommendations = [
    {
        title: "CLARIFY",
        text: "To help you get the best loan terms, I'll need to review your current financial statements, business registration, and recent bank statements. Do you have your 2023 and 2024 business income records available?"
    },
    {
        title: "PRODUCT RECOMMENDATION",
        text: "Based on your business type and expansion needs, I'd recommend our SME Business Growth Loan with flexible payment terms. We also have a special COVID Recovery Program that offers lower interest rates for businesses affected by the pandemic."
    },
    {
        title: "NEXT STEPS",
        text: "Would you like me to schedule a more detailed consultation call where we can review your application together? I can also prepare a pre-qualification assessment to give you a better idea of your loan options and terms."
    }
];

function renderRecPanel(recList) {
    return `
        <div class="rec-header">
          AI Message Recommendations
        </div>
        <div class="rec-list">
          ${recList.map(rec => `
            <div class="rec-item">
              <div class="rec-title">${rec.title}</div>
              <div class="rec-text">${rec.text}</div>
              <button class="rec-use-btn">Use</button>
            </div>
          `).join('')}
        </div>
      `;
}

// Render all panels using JS for reusability
document.getElementById('sidebar').innerHTML = renderSidebar(sidebarChats);
document.getElementById('chatPanel').innerHTML = renderChatPanel(chatPanelData);
document.getElementById('recPanel').innerHTML = renderRecPanel(recommendations);