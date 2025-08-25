export const autoChatCards = [
  { user: "User #1", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
  { user: "User #2", inquiry: "Loan Info Inquiry", percent: 50, time: "2 mins ago", status: "assigned", customerMsg: "Customer", agentMsg: "Broski" },
  { user: "User #3", inquiry: "Loan Info Inquiry", percent: 0, time: "2 mins ago", status: "assigned", customerMsg: "Customer", agentMsg: "" },
  { user: "User #4", inquiry: "Loan Info Inquiry", percent: 75, time: "3 mins ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
];

export const activeChatCards = [
  { user: "User #20", inquiry: "Loan Application", percent: 100, time: "just now", status: "assigned", customerMsg: "Customer", agentMsg: "AI Agent" },
  { user: "User #82", inquiry: "Account Closure", percent: 20, time: "1 min ago", status: "takeover", customerMsg: "Customer", agentMsg: "AI Agent" },
];

export const sidebarChats = [
  { user: "User #78", subtitle: "Loan Application", active: true },
  { user: "User #82", subtitle: "Account Closure", active: false },
  { user: "User #90", subtitle: "Loan Application", active: false }
];

export const chatTemplates = {
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
  }
};

export const recommendations = [
  { title: "CLARIFY", text: "To help you get the best loan terms, I'll need to review your financial statements..." },
  { title: "PRODUCT RECOMMENDATION", text: "Based on your business type, I'd recommend our SME Business Growth Loan..." },
  { title: "NEXT STEPS", text: "Would you like me to schedule a consultation call or prepare a pre-qualification?" }
];
