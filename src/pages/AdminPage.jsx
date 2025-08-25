import { useState, useEffect } from "react";
import "./AdminPage.css";
import { autoChatCards } from "../data";
import DashboardPanel from "../components/DashboardPanel";
import ChatLayout from "../components/ChatLayout";
import {
  sidebarChats as initialChats,
  chatTemplates,
  recommendations,
} from "../data";

export default function Admin() {
  const [activePanel, setActivePanel] = useState("panel1");
  const [cards, setCards] = useState([]);
  const [chats, setChats] = useState(initialChats);
  const activeUser = chats.find((c) => c.active)?.user || chats[0].user;

  const handleSidebarClick = (index) => {
    setChats(chats.map((c, i) => ({ ...c, active: i === index })));
  };

  // Load cards from localStorage on component mount, or use default data
  useEffect(() => {
    const savedCards = localStorage.getItem("chatCards");
    if (savedCards) {
      try {
        const parsedCards = JSON.parse(savedCards);
        setCards(parsedCards);
      } catch (error) {
        console.error("Error parsing saved cards:", error);
        // Fall back to default data if parsing fails
        setCards(
          autoChatCards.map((card, index) => ({
            ...card,
            id: card.id || index,
          }))
        );
      }
    } else {
      // Initialize with default data, ensuring each card has a unique ID
      const cardsWithIds = autoChatCards.map((card, index) => ({
        ...card,
        id: card.id || index,
      }));
      setCards(cardsWithIds);
    }
  }, []);

  // Save cards to localStorage whenever cards state changes
  useEffect(() => {
    if (cards.length > 0) {
      localStorage.setItem("chatCards", JSON.stringify(cards));
    }
  }, [cards]);

  const handleStatusUpdate = (cardId, newStatus) => {
    setCards((prevCards) =>
      prevCards.map((card) =>
        card.id === cardId || card.user === cardId
          ? { ...card, status: newStatus }
          : card
      )
    );
  };

  const handleTakeover = (card) => {
    // Any additional logic when takeover happens
    console.log(`Taking over chat for ${card?.user || "unknown user"}`);
    // Switch to active chats panel
    setActivePanel("panel2");
  };

  const getAssignedCards = () => {
    return cards.filter((card) => card.status === "assigned");
  };

  return (
    <div>
      {/* Header */}
      <div className="header">
        <span>AI Monitor</span>
        <div style={{ textAlign: "center" }}>
          <span style={{ color: "#fff" }}>Sakl</span>
          <span style={{ color: "#FDB415" }}>AI</span>
        </div>
        <span className="agent">Agent: John Doe</span>
      </div>

      {/* PANEL 1: Dashboard */}
      {activePanel === "panel1" && (
        <DashboardPanel
          cards={cards}
          onTakeover={handleTakeover}
          onStatusUpdate={handleStatusUpdate}
        />
      )}

      {/* PANEL 2: Active Chats */}
      {activePanel === "panel2" && (
        <ChatLayout
          chats={chats}
          onSidebarSelect={handleSidebarClick}
          activeChat={chatTemplates[activeUser]}
          recommendations={recommendations}
        />
      )}
    </div>
  );
}
