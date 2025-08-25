import { useState, useEffect } from "react";
import "./AdminPage.css";
import { autoChatCards } from "../data.js";
import DashboardPanel from "../components/DashboardPanel";

export default function Admin() {
  const [activePanel, setActivePanel] = useState("panel1");
  const [cards, setCards] = useState([]);

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
        <div id="panel2">
          <div className="tabs">
            <button className="tab" onClick={() => setActivePanel("panel1")}>
              Auto Chats
            </button>
            <button className="tab active">Active Chats</button>
          </div>
          <div className="active-chats">
            {getAssignedCards().length > 0 ? (
              <div className="grid" style={{ padding: "1rem" }}>
                {getAssignedCards().map((card) => (
                  <div key={card.id || card.user} className="assigned-card">
                    <h4>{card.user}</h4>
                    <p>
                      <strong>Type:</strong> {card.inquiry}
                    </p>
                    <p>
                      <strong>Time:</strong> {card.time}
                    </p>
                    <p>
                      <strong>Message:</strong> {card.customerMsg}
                    </p>
                    <p>
                      <strong>Status:</strong> Assigned to you
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ padding: "1rem" }}>
                No active chats assigned to you yet. Take over some chats from
                the Auto Chats panel.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
