import { useState } from "react";
import "./ClientPage.css";

export default function ClientPage() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I help you today?" },
  ]);
  const [input, setInput] = useState("");

  const sendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    // Simulated bot reply
    setTimeout(() => {
      setMessages((prev) => [...prev, { sender: "bot", text: "Thank you for your message!" }]);
    }, 800);
  };

  return (
    <div className="chat-container">
      <div className="chat-title">SaklAI Chat</div>
      <div className="chat-messages" id="chatMessages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
      </div>
      <form className="chat-input-area" onSubmit={sendMessage}>
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message..."
          autoComplete="off"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          required
        />
        <button type="submit" className="chat-send">
          Send
        </button>
      </form>
    </div>
  );
}
