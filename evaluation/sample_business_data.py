"""
Sample Business Data Generator for Testing Business Metrics
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

def generate_sample_business_data() -> Dict[str, Any]:
    """Generate realistic sample business data for testing business metrics."""
    
    # Generate session data for containment analysis
    sessions = []
    session_types = ["account_inquiry", "transaction_help", "product_info", "complaint", "technical_support"]
    
    for i in range(200):  # 200 sessions
        session = {
            "session_id": f"session_{i:04d}",
            "customer_id": f"customer_{random.randint(1000, 9999)}",
            "session_type": random.choice(session_types),
            "start_time": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "end_time": (datetime.now() - timedelta(days=random.randint(1, 30)) + timedelta(minutes=random.randint(2, 45))).isoformat(),
            "bot_handled": random.choice([True, False, True, True]),  # 75% bot handled
            "human_handoff": random.choice([False, False, False, True]),  # 25% handoff
            "resolution_status": random.choice(["resolved", "resolved", "escalated", "pending"]),
            "messages_count": random.randint(3, 20)
        }
        sessions.append(session)
    
    # Generate satisfaction data for CSAT analysis
    satisfaction_data = []
    for i in range(150):  # 150 satisfaction responses
        satisfaction = {
            "response_id": f"csat_{i:04d}",
            "session_id": f"session_{random.randint(0, 199):04d}",
            "rating": random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0],  # Weighted towards higher ratings
            "feedback": random.choice([
                "Very helpful bot", "Quick resolution", "Needed human help",
                "Good service", "Could be better", "Excellent support"
            ]),
            "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "would_recommend": random.choice([True, True, False, True])  # 75% would recommend
        }
        satisfaction_data.append(satisfaction)
    
    # Generate usage data for cost analysis
    usage_data = {
        "total_sessions": 200,
        "total_inferences": 3500,  # Multiple inferences per session
        "total_tokens_processed": 875000,  # ~250 tokens per inference average
        "model_costs": {
            "input_tokens": 650000,
            "output_tokens": 225000,
            "total_cost": 67.50  # $0.0675 per session average
        },
        "infrastructure_costs": {
            "compute_hours": 150,
            "storage_gb": 50,
            "bandwidth_gb": 25,
            "total_cost": 45.25
        },
        "operational_costs": {
            "human_agent_time_hours": 25,  # For handoffs
            "training_costs": 100.0,
            "maintenance_costs": 50.0,
            "total_cost": 150.0
        }
    }
    
    # Generate resolution data for FCR analysis
    resolution_data = []
    for i in range(180):  # 180 contact records
        resolution = {
            "contact_id": f"contact_{i:04d}",
            "customer_id": f"customer_{random.randint(1000, 9999)}",
            "issue_type": random.choice(["account_balance", "transaction_dispute", "product_inquiry", "technical_issue", "complaint"]),
            "first_contact_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "resolved_on_first_contact": random.choice([True, True, True, False]),  # 75% FCR
            "follow_up_required": random.choice([False, False, False, True]),  # 25% follow-up
            "resolution_date": (datetime.now() - timedelta(days=random.randint(0, 5))).isoformat(),
            "channel": random.choice(["chat", "email", "phone", "self_service"]),
            "complexity_score": random.randint(1, 5)
        }
        resolution_data.append(resolution)
    
    # Generate handoff events for handoff time analysis
    handoff_events = []
    for i in range(50):  # 50 handoff events
        handoff = {
            "handoff_id": f"handoff_{i:04d}",
            "session_id": f"session_{random.randint(0, 199):04d}",
            "bot_start_time": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "handoff_initiated_time": (datetime.now() - timedelta(days=random.randint(1, 30)) + timedelta(seconds=random.randint(30, 300))).isoformat(),
            "agent_response_time": (datetime.now() - timedelta(days=random.randint(1, 30)) + timedelta(seconds=random.randint(45, 400))).isoformat(),
            "handoff_reason": random.choice([
                "complex_inquiry", "customer_request", "bot_limitation", 
                "escalation_required", "technical_issue", "complaint"
            ]),
            "agent_id": f"agent_{random.randint(100, 199)}",
            "customer_wait_time_seconds": random.randint(15, 180),
            "total_handoff_time_seconds": random.randint(30, 120)
        }
        handoff_events.append(handoff)
    
    return {
        "sessions": sessions,
        "satisfaction_data": satisfaction_data,
        "usage_data": usage_data,
        "resolution_data": resolution_data,
        "handoff_events": handoff_events,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "data_period": "last_30_days",
            "total_customers": len(set([s["customer_id"] for s in sessions])),
            "business_hours": "9AM-5PM EST",
            "peak_hours": "10AM-2PM EST"
        }
    }

def save_sample_data(filename: str = "sample_business_data.json"):
    """Save sample business data to JSON file."""
    data = generate_sample_business_data()
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    return data

if __name__ == "__main__":
    # Generate and save sample data
    sample_data = save_sample_data("c:\\Users\\John Carlo\\saklAI\\evaluation\\sample_business_data.json")
    
    print("Sample Business Data Generated Successfully!")
    print(f"- Sessions: {len(sample_data['sessions'])}")
    print(f"- Satisfaction Responses: {len(sample_data['satisfaction_data'])}")
    print(f"- Resolution Records: {len(sample_data['resolution_data'])}")
    print(f"- Handoff Events: {len(sample_data['handoff_events'])}")
    print(f"- Total Customers: {sample_data['metadata']['total_customers']}")
    
    # Quick preview of business metrics
    from quantitative_metrics import BusinessMetrics
    
    business_calc = BusinessMetrics()
    
    # Calculate sample metrics
    containment = business_calc.calculate_bot_containment_rate(sample_data["sessions"])
    csat = business_calc.calculate_csat(sample_data["satisfaction_data"])
    cost = business_calc.calculate_cost_per_session(sample_data["usage_data"])
    fcr = business_calc.calculate_first_contact_resolution(sample_data["resolution_data"])
    handoff = business_calc.calculate_average_handoff_time(sample_data["handoff_events"])
    
    print("\n=== Sample Business Metrics Preview ===")
    print(f"Bot Containment Rate: {containment.get('bot_containment_rate', 0):.1f}%")
    print(f"CSAT Score: {csat.get('csat_score', 0):.1f}%")
    print(f"Cost per Session: ${cost.get('cost_per_session', 0):.3f}")
    print(f"First Contact Resolution: {fcr.get('fcr_rate', 0):.1f}%")
    print(f"Average Handoff Time: {handoff.get('average_handoff_time', 0):.1f}s")
