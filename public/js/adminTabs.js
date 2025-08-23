// Data for both tabs
const autoChatCards = [
    {
        user: "User #1",
        inquiry: "Loan Info Inquiry",
        percent: 50,
        time: "2 mins ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #2",
        inquiry: "Loan Info Inquiry",
        percent: 50,
        time: "2 mins ago",
        status: "assigned",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #3",
        inquiry: "Loan Info Inquiry",
        percent: 0,
        time: "2 mins ago",
        status: "complex",
        customerMsg: "Customer",
        agentMsg: ""
    },
    {
        user: "User #4",
        inquiry: "Loan Info Inquiry",
        percent: 50,
        time: "2 mins ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #5",
        inquiry: "Loan Info Inquiry",
        percent: 50,
        time: "2 mins ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #6",
        inquiry: "Loan Info Inquiry",
        percent: 50,
        time: "2 mins ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    }
];

const activeChatCards = [
    {
        user: "User #20",
        inquiry: "Loan Application",
        percent: 100,
        time: "just now",
        status: "assigned",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #82",
        inquiry: "Account Closure",
        percent: 20,
        time: "1 min ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    },
    {
        user: "User #90",
        inquiry: "Loan Application",
        percent: 0,
        time: "2 mins ago",
        status: "complex",
        customerMsg: "Customer",
        agentMsg: ""
    },
    {
        user: "User #45",
        inquiry: "Loan Info Inquiry",
        percent: 75,
        time: "3 mins ago",
        status: "takeover",
        customerMsg: "Customer",
        agentMsg: "AI Agent"
    }
];

// Render function for reusable JS card
function renderCard(card) {
    const isYellow = card.status === "assigned" || card.status === "complex";
    const percentClass = card.percent === 0 ? "percent zero" :
        isYellow ? "percent yellow" : card.percent < 100 ? "percent low" : "percent";

    return `
    <div class="chat-card${isYellow ? ' yellow' : ''}">
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
            : `<div class="chat-bubble agent">${card.agentMsg}</div>`
        }
        </div>
      </div>
      <div class="chat-footer">
        ${card.status === "takeover"
            ? `<button class="btn takeover">Take Over</button>`
            : `<button class="btn assigned" disabled>Assigned</button>`
        }
      </div>
    </div>
  `;
}

function renderGrid(cards) {
    document.getElementById("chatGrid").innerHTML = cards.map(renderCard).join("");
}

// Initial render
renderGrid(autoChatCards);

// Tab click logic
document.getElementById("autoTab").onclick = function () {
    this.classList.add("active");
    document.getElementById("activeTab").classList.remove("active");
    document.getElementById("activeTab").classList.add("secondary");
    this.classList.remove("secondary");
    renderGrid(autoChatCards);
};

document.getElementById("activeTab").onclick = function () {
    this.classList.add("active");
    document.getElementById("autoTab").classList.remove("active");
    document.getElementById("autoTab").classList.add("secondary");
    this.classList.remove("secondary");
    renderGrid(activeChatCards);
};