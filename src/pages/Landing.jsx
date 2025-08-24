import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="container">
      <div className="logo">
        Sakl<span className="ai">AI</span>
      </div>
      <div className="subtitle">
        A Sentiment-Aware AI Agent for Hyper-Personalized Customer Experience
      </div>
      <Link to="/admin">
        <button className="button">Admin</button>
      </Link>
      <Link to="/chat">
        <button className="button">Customer</button>
      </Link>
    </div>
  );
}
