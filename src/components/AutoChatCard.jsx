import "./AutoChatCard.css";

export default function ChatCard({ card, onTakeover, onTakeOver, onStatusUpdate }) {
  const isYellow = ["assigned", "complex"].includes(card.status);

  const handleTakeover = () => {
    // Update the card status to "assigned"
    if (onStatusUpdate) {
      onStatusUpdate(card.id || card.user, "assigned");
    }
    
    // Call the original onTakeover/onTakeOver if needed for other logic
    if (onTakeover) {
      onTakeover(card);
    }
    if (onTakeOver) {
      onTakeOver(card);
    }
  };

  return (
    <div className={`chat-card ${isYellow ? "yellow" : ""}`}>
      <div>
        <div className="chat-header">
          <div>
            <div className="chat-user">{card.user}</div>
            <div className="chat-type">{card.inquiry}</div>
          </div>
          <div className="chat-meta">
            <span className="time">{card.time}</span>
          </div>
        </div>
        <div className="chat-content">
          <div className="chat-bubble">{card.customerMsg}</div>
          {card.status === "complex" ? (
            <div className="chat-bubble yellow">
              Complex Case Detected. Assigning to human agent...
            </div>
          ) : (
            <div className="chat-bubble-wrapper">
              <div className="chat-label">AI Agent</div>
              <div className="chat-bubble agent">{card.agentMsg}</div>
            </div>
          )}
        </div>
      </div>
      <div className="chat-footer">
        {card.status === "takeover" ? (
          <button className="btn takeover" onClick={handleTakeover}>
            Take Over
          </button>
        ) : (
          <button className="btn assigned" disabled>
            Assigned
          </button>
        )}
      </div>
    </div>
  );
}