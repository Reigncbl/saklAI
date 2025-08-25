import "./Chat.css";
export default function ChatPanel({ chat }) {
  return (
    <div className="chat-panel">
      <div className="chat-header">{chat.name}</div>
      <div className="chat-messages">
        {chat.messages.map((m, idx) => (
          <div key={idx} className={`chat-row${m.agent ? " agent" : ""}`}>
            {m.label && (
              <div className={`chat-label${m.agent ? " agent" : ""}`}>
                {m.label}
              </div>
            )}
            <div
              className={`chat-bubble${m.agent ? " agent" : ""}${
                m.yellow ? " yellow" : ""
              }`}
            >
              {m.bubble}
            </div>
          </div>
        ))}
      </div>
      <form
        className="chat-input-area"
        onSubmit={(e) => {
          e.preventDefault();
        }}
      >
        <input
          type="text"
          placeholder="Type your message here..."
          className="chat-input"
        />
        <button type="submit" className="chat-send">
          Send
        </button>
      </form>
    </div>
  );
}
