import ChatCard from "./AutoChatCard";
import "../pages/AdminPage.css";

export default function DashboardPanel({ cards, onTakeover, onStatusUpdate }) {
  return (
    <div id="panel1">
      <div className="tabs">
        <button className="tab active">Auto Chats</button>
        <button className="tab" onClick={onTakeover}>
          Active Chats
        </button>
      </div>
      <div id="admin-container">
        <div className="grid" id="chatGrid">
          {cards.map((card, i) => (
            <ChatCard 
              key={card.id || card.user || i} 
              card={card} 
              onTakeover={onTakeover}
              onStatusUpdate={onStatusUpdate}
            />
          ))}
        </div>
      </div>
    </div>
  );
}