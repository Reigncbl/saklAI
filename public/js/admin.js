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

// Global variable to store current AI recommendations
let currentAIRecommendations = [];

// Function to fetch AI recommendations for a specific user
async function fetchAIRecommendations(userId) {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/recommendations/${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    
    const data = await response.json();
    console.log("AI Recommendations received:", data);
    
    if (data.status === 'success' && data.recommendations) {
      // Format recommendations for the UI
      currentAIRecommendations = data.recommendations.map(rec => ({
        title: rec.category || rec.type || "GENERAL", // Handle both category and type fields
        text: rec.message,
        reasoning: rec.reasoning || ''
      }));
      
      // Update the recommendations panel
      updateRecommendationsPanel();
      return currentAIRecommendations;
    } else {
      console.warn("Failed to get AI recommendations:", data.message);
      return recommendations; // Fallback to static recommendations
    }
  } catch (error) {
    console.error("Error fetching AI recommendations:", error);
    return recommendations; // Fallback to static recommendations
  }
}

// Function to update the recommendations panel
function updateRecommendationsPanel() {
  const recPanel = el("recPanel");
  if (recPanel) {
    const recommendationsToShow = currentAIRecommendations.length > 0 ? currentAIRecommendations : recommendations;
    recPanel.innerHTML = renderRecPanel(recommendationsToShow);
  }
}

// Function to refresh recommendations for the currently selected user
async function refreshRecommendations() {
  const activeChat = currentActiveChatSessions.find(c => c.active);
  if (activeChat && activeChat.fullData && activeChat.fullData.status === "assigned") {
    // The user field already contains the correct user_id format (customer_xxxxx)
    const userId = activeChat.user;
    console.log("Fetching recommendations for user ID:", userId);
    await fetchAIRecommendations(userId);
  }
}

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

  // Check if agent was requested by looking for agent request messages
  let isAgentRequested = false;
  if (Array.isArray(card.history)) {
    isAgentRequested = card.history.some(m => 
      m.content && (
        m.content.includes('AGENT REQUESTED') || 
        m.content.includes('Customer has requested to speak with a human agent')
      )
    );
  }

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
    <div class="chat-card${isYellow ? " yellow" : ""}${isAgentRequested ? " agent-requested" : ""}">
      <div>
        <div class="chat-header">
          <div>
            <div class="chat-user">${card.user}</div>
            <div class="chat-type">${card.inquiry}</div>
          </div>
          <div class="chat-meta">
            <span class="time">${card.time}</span>
            <span class="status-badge status-${card.status}">${card.status?.toUpperCase() || 'ACTIVE'}</span>
          </div>
        </div>
        <div class="chat-content">
          ${chatBubbles}
        </div>
      </div>
      <div class="chat-footer">
        <div class="footer-info">
          <span class="message-count">${card.message_count || 0} messages</span>
        </div>
        <button class="btn takeover" data-user="${card.user}">Take Over</button>
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

    // Show all chats in Auto Chats panel for monitoring purposes
    const availableChats = chats; // Show all chats instead of filtering
    console.log("All chats for Auto Chats panel:", availableChats);

    // Map backend data to card format expected by renderGrid
    const cards = availableChats.map(c => {
      // Get the last customer message for inquiry summary
      const lastCustomerMessage = c.history?.filter(m => m.role === 'user').pop();
      const inquiryText = lastCustomerMessage?.content?.substring(0, 50) + (lastCustomerMessage?.content?.length > 50 ? '...' : '') || "Customer Inquiry";
      
      return {
        user: c.user_id,
        inquiry: inquiryText,
        time: c.last_timestamp ? new Date(c.last_timestamp).toLocaleTimeString() : "Unknown",
        status: c.status || "active",
        history: c.history || [],
        message_count: c.message_count || 0
      };
    });

    renderGrid(cards);
  } catch (err) {
    console.error("Failed to fetch active chats:", err);
    // Optionally show a message in the grid
    el("chatGrid").innerHTML = `<div class='error'>Failed to load active chats.</div>`;
  }
}


// PANEL 2 - Active Chat Management

async function fetchActiveChatSessions() {
  try {
    const response = await fetch(`${API_BASE_URL}/chat/active`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const chats = await response.json();
    console.log("Active chat sessions:", chats);

    // Filter only chats with "assigned" status for Panel 2
    const assignedChats = chats.filter(chat => chat.status === "assigned");
    console.log("Filtered assigned chats:", assignedChats);

    // Map to sidebar format
    const sidebarData = assignedChats.map((chat, index) => ({
      user: chat.user_id,
      subtitle: chat.last_message ? chat.last_message.substring(0, 30) + '...' : 'Active Chat',
      active: index === 0, // First one active by default
      fullData: chat // Store full chat data
    }));

    return sidebarData;
  } catch (err) {
    console.error("Failed to fetch active chat sessions:", err);
    // Return default data if API fails (filter only assigned status)
    return sidebarChats.filter(chat => chat.status === "assigned" || !chat.status);
  }
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
  // Filter messages if we have history data with status information
  let filteredMessages = chat.messages;
  if (chat.fullData && chat.fullData.status === "assigned" && chat.fullData.history) {
    // Only show messages from assigned chats
    filteredMessages = chat.fullData.history.map(msg => {
      const messageContent = msg.content || (msg.response && msg.response.suggestions ? msg.response.suggestions[0]?.suggestion : '');
      const isAgentRequest = messageContent.includes('AGENT REQUESTED') || messageContent.includes('Customer has requested to speak with a human agent');
      
      return {
        label: msg.role === 'user' ? 'Customer' : (msg.role === 'assistant' || msg.role === 'agent' ? 'Agent' : 'System'),
        bubble: messageContent,
        agent: msg.role === 'assistant' || msg.role === 'agent',
        yellow: isAgentRequest || msg.role === 'system', // Highlight agent requests and system messages
        urgent: isAgentRequest // Special flag for agent requests
      };
    });
  }

  return `
    <div class="chat-header">${chat.name || chat.user || 'Customer Chat'}</div>
    <div class="chat-messages">
      ${filteredMessages && filteredMessages.length > 0 ? filteredMessages.map(m => `
        <div class="chat-row${m.agent ? " agent" : ""}${m.urgent ? " urgent" : ""}">
          ${m.label ? `<div class="chat-label${m.agent ? " agent" : ""}">${m.label}</div>` : ""}
          ${m.bubble ? `<div class="chat-bubble${m.agent ? " agent" : ""}${m.yellow ? " yellow" : ""}${m.urgent ? " urgent" : ""}">${m.bubble}</div>` : ""}
        </div>`).join("") : '<div class="chat-row"><div class="chat-bubble">No messages available for assigned chats...</div></div>'}
    </div>
    <form class="chat-input-area" onsubmit="event.preventDefault(); sendMessage(this);">
      <input type="text" class="chat-input" placeholder="Type your message here..." autocomplete="off" />
      <button type="submit" class="chat-send">Send</button>
    </form>`;
}

function renderRecPanel(list) {
  return `
    <div class="rec-header">
      AI Message Recommendations
      <button class="rec-refresh-btn" onclick="refreshRecommendations()" title="Get fresh AI recommendations">
        🔄
      </button>
    </div>
    <div class="rec-list">
      ${list.map(r => `
        <div class="rec-item">
          <div class="rec-title">${r.title}</div>
          <div class="rec-text">${r.text}</div>
          ${r.reasoning ? `<div class="rec-reasoning">💡 ${r.reasoning}</div>` : ''}
          <button class="rec-use-btn" type="button" onclick="useRecommendation('${r.text.replace(/'/g, "\\'")}')">Use</button>
        </div>`).join("")}
    </div>`;
}

// Global variable to store current active chat sessions
let currentActiveChatSessions = [];

async function mountPanel2() {
  // Fetch fresh active chat sessions (filtered for assigned status)
  currentActiveChatSessions = await fetchActiveChatSessions();

  // Render sidebar with assigned sessions only
  el("sidebar").innerHTML = renderSidebar(currentActiveChatSessions);

  // Render the first assigned chat or show no chats message
  const activeChat = currentActiveChatSessions.find(c => c.active);
  if (activeChat && activeChat.fullData && activeChat.fullData.status === "assigned") {
    // Convert backend data to chat format for assigned chats only
    const chatData = {
      name: activeChat.user,
      user: activeChat.user,
      fullData: activeChat.fullData, // Pass full data for filtering
      messages: activeChat.fullData.history ? activeChat.fullData.history.map(msg => ({
        label: msg.role === 'user' ? 'Customer' : (msg.role === 'assistant' || msg.role === 'agent' ? 'Agent' : 'System'),
        bubble: msg.content || (msg.response && msg.response.suggestions ? msg.response.suggestions[0]?.suggestion : ''),
        agent: msg.role === 'assistant' || msg.role === 'agent'
      })) : []
    };
    el("chatPanel").innerHTML = renderChatPanel(chatData);
    
    // Fetch AI recommendations for the active chat
    const userId = activeChat.user.replace('User #', 'customer_');
    fetchAIRecommendations(userId);
  } else if (currentActiveChatSessions.length === 0) {
    // No assigned chats available
    el("chatPanel").innerHTML = renderChatPanel({
      name: 'No Assigned Chats',
      messages: [{ label: "System", bubble: "No assigned chats available. Only chats with 'assigned' status are shown here.", agent: false }]
    });
    // Reset to default recommendations
    currentAIRecommendations = [];
    updateRecommendationsPanel();
  } else {
    // Fallback to first available assigned chat
    const firstAssigned = currentActiveChatSessions[0];
    if (firstAssigned && firstAssigned.fullData) {
      const chatData = {
        name: firstAssigned.user,
        user: firstAssigned.user,
        fullData: firstAssigned.fullData,
        messages: firstAssigned.fullData.history ? firstAssigned.fullData.history.map(msg => ({
          label: msg.role === 'user' ? 'Customer' : (msg.role === 'assistant' || msg.role === 'agent' ? 'Agent' : 'System'),
          bubble: msg.content || (msg.response && msg.response.suggestions ? msg.response.suggestions[0]?.suggestion : ''),
          agent: msg.role === 'assistant' || msg.role === 'agent'
        })) : []
      };
      el("chatPanel").innerHTML = renderChatPanel(chatData);
      
      // Fetch AI recommendations for the first chat
      const userId = firstAssigned.user.replace('User #', 'customer_');
      fetchAIRecommendations(userId);
    }
  }

  // Render recommendations (will show default ones initially)
  const recommendationsToShow = currentAIRecommendations.length > 0 ? currentAIRecommendations : recommendations;
  el("recPanel").innerHTML = renderRecPanel(recommendationsToShow);
}

// Message sending functionality
async function sendMessage(form) {
  const input = form.querySelector('.chat-input');
  const message = input.value.trim();
  if (!message) return;

  // Get current active chat
  const activeChat = currentActiveChatSessions.find(c => c.active);
  if (!activeChat) return;

  // Disable input while sending
  input.disabled = true;
  const submitBtn = form.querySelector('.chat-send');
  const originalBtnText = submitBtn.textContent;
  submitBtn.textContent = 'Sending...';
  submitBtn.disabled = true;

  try {
    // Add message to chat display immediately
    const messagesContainer = form.previousElementSibling;
    const messageHtml = `
      <div class="chat-row agent">
        <div class="chat-label agent">Agent</div>
        <div class="chat-bubble agent">${message}</div>
      </div>`;
    messagesContainer.insertAdjacentHTML('beforeend', messageHtml);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Clear input
    input.value = '';

    // Send to backend API to save in JSON file
    const response = await fetch(`${API_BASE_URL}/chat/message/${activeChat.user}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: message })  // Send as object with message field
    });

    if (!response.ok) {
      throw new Error(`Failed to save message: ${response.status}`);
    }

    const result = await response.json();
    console.log(`Message saved to ${activeChat.user} chat history:`, result);

    // Update the local chat data to include the new message
    if (activeChat.fullData && activeChat.fullData.history) {
      activeChat.fullData.history.push({
        timestamp: result.timestamp,
        role: "agent",
        content: message,
        response: null,
        template_used: null,
        processing_method: "human_agent"
      });
    }

  } catch (error) {
    console.error('Error sending message:', error);
    alert('Failed to send message. Please try again.');

    // Remove the message from display if it failed to save
    const lastMessage = messagesContainer.lastElementChild;
    if (lastMessage && lastMessage.classList.contains('agent')) {
      lastMessage.remove();
    }
  } finally {
    // Re-enable input
    input.disabled = false;
    submitBtn.textContent = originalBtnText;
    submitBtn.disabled = false;
    input.focus();
  }
}

// Use recommendation functionality
async function useRecommendation(text) {
  const chatInput = document.querySelector('.chat-input');
  if (chatInput) {
    chatInput.value = text;
    chatInput.focus();
    
    // Re-enable bot when admin uses AI recommendations
    const activeChat = currentActiveChatSessions.find(c => c.active);
    if (activeChat && activeChat.user) {
      try {
        const response = await fetch(`${API_BASE_URL}/chat/status/${activeChat.user}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ 
            status: 'assigned',  // Keep assigned status
            bot_enabled: true    // Re-enable bot responses
          })
        });
        
        if (response.ok) {
          console.log('Bot re-enabled for user:', activeChat.user);
          
          // Add a subtle visual indicator that bot is re-enabled
          const button = event.target;
          const originalText = button.textContent;
          button.textContent = 'Bot Enabled ✓';
          button.style.background = '#22c55e';
          setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
          }, 2000);
        }
      } catch (error) {
        console.error('Error re-enabling bot:', error);
      }
    }
  }
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

el("autoTab").onclick = () => {
  showPanel(1);
  fetchActiveChats();
};

el("activeTab").onclick = () => {
  showPanel(2);
  mountPanel2();
};

// Auto-refresh functionality - REMOVED
// Replaced with manual refresh button

// Manual refresh function
function refreshCurrentPanel() {
  const panel1Visible = !el("panel1").classList.contains("hidden");
  const panel2Visible = !el("panel2").classList.contains("hidden");

  // Show loading state on button
  const refreshBtn = el("refreshBtn");
  const originalText = refreshBtn.innerHTML;
  refreshBtn.innerHTML = "🔄 Refreshing...";
  refreshBtn.disabled = true;

  if (panel1Visible) {
    console.log("Refreshing Panel 1 data...");
    fetchActiveChats().finally(() => {
      // Restore button state
      refreshBtn.innerHTML = originalText;
      refreshBtn.disabled = false;
    });
  } else if (panel2Visible) {
    console.log("Refreshing Panel 2 data...");
    // Preserve current selection when refreshing Panel 2
    const currentActiveIndex = currentActiveChatSessions.findIndex(c => c.active);
    mountPanel2().then(() => {
      // Restore selection if possible
      if (currentActiveIndex >= 0 && currentActiveIndex < currentActiveChatSessions.length) {
        currentActiveChatSessions.forEach((c, i) => c.active = i === currentActiveIndex);
        el("sidebar").innerHTML = renderSidebar(currentActiveChatSessions);
      }
    }).finally(() => {
      // Restore button state
      refreshBtn.innerHTML = originalText;
      refreshBtn.disabled = false;
    });
  } else {
    // No panel visible, restore button immediately
    refreshBtn.innerHTML = originalText;
    refreshBtn.disabled = false;
  }
}

// Start initial load when page loads
window.addEventListener("load", () => {
  fetchActiveChats();
});

document.addEventListener("click", (e) => {
  // Handle takeover button clicks
  if (e.target.closest(".btn.takeover")) {
    const btn = e.target.closest(".btn.takeover");
    const userId = btn.getAttribute("data-user");

    // Show loading state on button
    const originalText = btn.textContent;
    btn.textContent = "Taking over...";
    btn.disabled = true;

    // Call backend to set status to 'assigned'
    fetch(`${API_BASE_URL}/chat/status/${userId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "assigned" })
    })
      .then(res => {
        if (!res.ok) throw new Error("Failed to assign chat");
        return res.json();
      })
      .then(result => {
        console.log("Chat assigned successfully:", result);

        // Switch to panel 2 first
        showPanel(2);

        // Load Panel 2 data and navigate to the specific conversation
        mountPanel2().then(() => {
          // Find the conversation in the current active chat sessions
          const conversationIndex = currentActiveChatSessions.findIndex(chat => chat.user === userId);

          if (conversationIndex >= 0) {
            // Set this conversation as active
            currentActiveChatSessions.forEach((c, i) => c.active = i === conversationIndex);

            // Re-render sidebar with the new active selection
            el("sidebar").innerHTML = renderSidebar(currentActiveChatSessions);

            // Update chat panel with the selected conversation
            const selectedChat = currentActiveChatSessions[conversationIndex];
            if (selectedChat && selectedChat.fullData) {
              const chatData = {
                name: selectedChat.user,
                user: selectedChat.user,
                fullData: selectedChat.fullData,
                messages: selectedChat.fullData.history ? selectedChat.fullData.history.map(msg => ({
                  label: msg.role === 'user' ? 'Customer' : (msg.role === 'assistant' || msg.role === 'agent' ? 'Agent' : 'System'),
                  bubble: msg.content || (msg.response && msg.response.suggestions ? msg.response.suggestions[0]?.suggestion : ''),
                  agent: msg.role === 'assistant' || msg.role === 'agent'
                })) : []
              };
              el("chatPanel").innerHTML = renderChatPanel(chatData);
            }

            console.log(`Navigated to conversation with ${userId}`);
          } else {
            console.warn(`Conversation ${userId} not found in assigned chats`);
            // If not found, just show the first available chat
          }
        });

        // Refresh panel 1 to remove the taken over chat
        fetchActiveChats();
      })
      .catch(err => {
        console.error("Takeover failed:", err);
        alert("Failed to assign chat: " + err.message);

        // Restore button state on error
        btn.textContent = originalText;
        btn.disabled = false;
      });
  }

  // Handle sidebar chat selection
  const li = e.target.closest("#sidebar .sidebar-list li");
  if (li) {
    const index = parseInt(li.dataset.index);

    // Update active state
    currentActiveChatSessions.forEach((c, i) => c.active = i === index);

    // Re-render sidebar
    el("sidebar").innerHTML = renderSidebar(currentActiveChatSessions);

    // Update chat panel with selected assigned chat
    const selectedChat = currentActiveChatSessions[index];
    if (selectedChat && selectedChat.fullData && selectedChat.fullData.status === "assigned") {
      const chatData = {
        name: selectedChat.user,
        user: selectedChat.user,
        fullData: selectedChat.fullData, // Pass full data for filtering
        messages: selectedChat.fullData.history ? selectedChat.fullData.history.map(msg => ({
          label: msg.role === 'user' ? 'Customer' : (msg.role === 'assistant' || msg.role === 'agent' ? 'Agent' : 'System'),
          bubble: msg.content || (msg.response && msg.response.suggestions ? msg.response.suggestions[0]?.suggestion : ''),
          agent: msg.role === 'assistant' || msg.role === 'agent'
        })) : []
      };
      el("chatPanel").innerHTML = renderChatPanel(chatData);
      
      // Fetch AI recommendations for this user
      const userId = selectedChat.user.replace('User #', 'customer_');
      fetchAIRecommendations(userId);
    } else {
      // Show message for non-assigned chats
      const chatData = {
        name: selectedChat ? selectedChat.user : 'Unknown User',
        messages: [{ label: "System", bubble: "This chat is not assigned to human agents. Only assigned chats are shown in this panel.", agent: false }]
      };
      el("chatPanel").innerHTML = renderChatPanel(chatData);
      
      // Reset to default recommendations
      currentAIRecommendations = [];
      updateRecommendationsPanel();
    }
  }
});

// Initialize the admin panel when the page loads
document.addEventListener('DOMContentLoaded', function() {
  console.log('Admin panel loading...');
  
  // Set initial active tab
  const autoTab = document.getElementById('autoTab');
  const activeTab = document.getElementById('activeTab');
  
  if (autoTab) {
    autoTab.classList.add('active');
    // Load the Auto Chats panel (Panel 1) by default
    fetchActiveChats();
  }
  
  if (activeTab) {
    activeTab.classList.remove('active');
  }
  
  // Set up automatic refresh every 5 seconds
  setInterval(() => {
    if (document.getElementById('autoTab').classList.contains('active')) {
      fetchActiveChats();
    } else if (document.getElementById('activeTab').classList.contains('active')) {
      mountPanel2();
    }
  }, 5000);
  
  console.log('Admin panel initialized');
});
