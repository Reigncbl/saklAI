import "./Chat.css";
export default function RecommendationsPanel({ recommendations }) {
  return (
    <div className="rec-panel">
      <div className="rec-header">AI Message Recommendations</div>
      <div className="rec-list">
        {recommendations.map((r, i) => (
          <div key={i} className="rec-item">
            <div className="rec-title">{r.title}</div>
            <div className="rec-text">{r.text}</div>
            <button className="rec-use-btn">Use</button>
          </div>
        ))}
      </div>
    </div>
  );
}
