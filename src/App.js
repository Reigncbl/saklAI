import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import AdminPage from "./pages/AdminPage";
import ClientPage from "./pages/ClientPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="/chat" element={<ClientPage />} />
      </Routes>
    </Router>
  );
}

export default App;
