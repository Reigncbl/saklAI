import Sidebar from "./ChatSidebar";
import ChatPanel from "./ChatPanel";
import RecommendationsPanel from "./ChatReco";

export default function ChatLayout({
  chats,
  onSidebarSelect,
  activeChat,
  recommendations,
  onTakeover,
}) {
  return (
    <div id="panel2">
      <div className="tabs">
        <button className="tab active">Auto Chats</button>
        <button className="tab" onClick={onTakeover}>
          Active Chats
        </button>
      </div>
      <div className="panel2 grid grid-cols-3 gap-4">
        <Sidebar chats={chats} onSelect={onSidebarSelect} />
        <ChatPanel chat={activeChat} />
        <RecommendationsPanel recommendations={recommendations} />
      </div>
    </div>
  );
}
