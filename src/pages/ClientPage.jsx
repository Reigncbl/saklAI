import React, { useEffect, useState } from "react";
import "./ClientPage.css";

const ClientPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [userId] = useState(
    "customer_" + Math.random().toString(36).substr(2, 9)
  );
  const [isTyping, setIsTyping] = useState(false);

  // Determine API base URL
  const API_BASE_URL = (() => {
    if (window.location.protocol === "file:") {
      return "http://localhost:8000";
    }
    if (
      window.location.hostname === "localhost" &&
      window.location.port !== "8000"
    ) {
      return "http://localhost:8000";
    }
    return "";
  })();

  // Load chat history on mount
  useEffect(() => {
    const loadChatHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/chat/history/${userId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.history && data.history.length > 0) {
            setMessages(
              data.history.map((entry) => ({
                sender: entry.type === "user" ? "user" : "bot",
                text: entry.message,
              }))
            );
          } else {
            setMessages([
              { sender: "bot", text: "Hello! How can I help you today?" },
            ]);
          }
        } else {
          setMessages([
            { sender: "bot", text: "Hello! How can I help you today?" },
          ]);
        }
      } catch (err) {
        setMessages([
          { sender: "bot", text: "Hello! How can I help you today?" },
        ]);
      }
    };

    loadChatHistory();
  }, [API_BASE_URL, userId]);

  // Send message
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const msg = input.trim();
    setMessages((prev) => [...prev, { sender: "user", text: msg }]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch(`${API_BASE_URL}/rag/suggestions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          message: msg,
          prompt_type: "auto",
          include_context: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! ${response.status}`);
      }

      const data = await response.json();
      setIsTyping(false);

      if (data.status === "success" && data.suggestions?.length > 0) {
        // main suggestion
        const suggestion = data.suggestions[0];
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text:
              suggestion.suggestion ||
              suggestion.analysis ||
              "I received your message.",
          },
        ]);

        // extra suggestions
        if (data.suggestions.length > 1) {
          data.suggestions.slice(1).forEach((extra) => {
            if (extra.suggestion || extra.analysis) {
              setMessages((prev) => [
                ...prev,
                { sender: "bot", text: extra.suggestion || extra.analysis },
              ]);
            }
          });
        }
      } else {
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: "Thank you for your message. How else can I help you?",
          },
        ]);
      }
    } catch (err) {
      setIsTyping(false);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "Sorry, I'm having trouble connecting right now. Please try again later.",
        },
      ]);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-title">SaklAI Chat</div>
      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.sender}`}>
            {m.text}
          </div>
        ))}
        {isTyping && (
          <div className="message bot typing-indicator">
            <span>SaklAI is typing</span>
            <span className="dots">...</span>
          </div>
        )}
      </div>
      <form className="chat-input-area" onSubmit={sendMessage}>
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          autoComplete="off"
          required
        />
        <button type="submit" className="chat-send">
          Send
        </button>
      </form>
    </div>
  );
};
export default ClientPage;
