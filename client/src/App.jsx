import Home from "./page/index.jsx";
//Strawberry-Insight-AI
//disease-detection-service
//advisory-service
//gateway-service
//strawberry-validation-service
//source .venv/Scripts/activate & .\venv\Scripts\Activate.ps1 (power shell)
//uvicorn main:app --reload --port 8000
//uvicorn main:app --reload --host 0.0.0.0 --port 8002
function App() {
  return (
    <div className="App">
      <Home />
    </div>
  );
}

export default App;