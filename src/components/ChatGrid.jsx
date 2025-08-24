import ChatCard from "./AutoChatCard";
import "../pages/AdminPage.css";

export default function ChatGrid({ cards, onTakeOver, onStatusUpdate }) {
  return (
    <div className="grid">
      {cards.map((card) => (
        <ChatCard 
          key={card.id || card.user} 
          card={card} 
          onTakeOver={onTakeOver}
          onStatusUpdate={onStatusUpdate}
        />
      ))}
    </div>
  );
}