import "./Chat.css";
export default function Sidebar({ chats, onSelect }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">Chats</div>
      <ul className="sidebar-list">
        {chats.map((c, i) => (
          <li
            key={i}
            className={c.active ? "active" : ""}
            onClick={() => onSelect(i)}
          >
            {c.user}
            {c.subtitle && <div className="subtitle">{c.subtitle}</div>}
          </li>
        ))}
      </ul>
    </aside>
  );
}
