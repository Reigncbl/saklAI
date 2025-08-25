
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi

# 1. Data Initialization
# This data structure will be used to generate all plots
data = {
    'technical_performance': {
        'metric': ['Response Time (s)', 'Throughput (QPS)'],
        'saklai': [1.2, 276],
        'benchmark': [2.5, 180],
        'source': [5]
    },
    'rag_metrics': {
        'Precision@K': 0.88,
        'Recall@K': 0.82,
        'MRR': 0.79,
        'Context Relevance': 0.91,
        'Answer Faithfulness': 0.94
    },
    'customer_business': {
        'metric': ['Customer Satisfaction', 'Query Success Rate', 'Resolution Time', 'User Engagement'],
        'saklai': [0.89, 0.84, 0.78, 3.2],
        'benchmark': [0.75, 0.70, 0.70, 5.0],
        'source': [19, 20, 21, 22]
    },
    'cost_efficiency': {
        'model': ['SaklAI', 'GPT-4', 'Claude-3', 'Gemini-Pro'],
        'throughput_qps': [276, 150, 200, 180], # Mock throughput for comparison
        'cost_per_m_tokens': [0.015, 0.05, 0.15, 0.35],
        'model_capacity': [100, 150, 120, 140]  # Mock values for bubble size
    }
}

# 2. Visualization Modules

# Line/Bar Chart (Response Time vs. Throughput)
def plot_response_time_throughput():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Throughput Bar Chart
    throughput_data = [data['technical_performance']['saklai'][1]]
    labels = ['SaklAI']
    ax1.bar(labels, throughput_data, color='skyblue', label='Throughput (QPS)')
    ax1.set_ylabel('Throughput (QPS)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_title('Response Time vs. Throughput')
    
    # Response Time Line Chart with SLA
    ax2 = ax1.twinx()
    response_time_data = [data['technical_performance']['saklai'][0]]
    ax2.plot(labels, response_time_data, color='red', marker='o', label='Response Time (s)')
    ax2.set_ylabel('Response Time (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # SLA line
    ax2.axhline(y=3, color='orange', linestyle='--', label='SLA Target (<3s)')
    
    fig.tight_layout()
    plt.show()

# Radar Chart (RAG Metrics)
def plot_rag_metrics():
    labels = list(data['rag_metrics'].keys())
    values = list(data['rag_metrics'].values())
    
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, color='grey', size=12)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    ax.set_ylim(0, 1)
    
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='SaklAI Metrics')
    ax.fill(angles, values, 'b', alpha=0.25)
    
    plt.title('RAG / Knowledge Base Metrics', size=16, color='black', y=1.1)
    plt.show()

# Vertical Bar Chart (Customer & Business Metrics)
def plot_customer_business_metrics():
    df = pd.DataFrame(data['customer_business'])
    
    df_melted = df.melt(id_vars='metric', var_name='Type', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='metric', y='Value', hue='Type', data=df_melted)
    plt.title('Customer & Business Metrics vs. Targets')
    plt.ylabel('Score / Value')
    plt.show()

# Bubble Chart (Cost & Efficiency)
def plot_cost_efficiency_bubble():
    df = pd.DataFrame(data['cost_efficiency'])
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='throughput_qps', y='cost_per_m_tokens', size='model_capacity', hue='model',
                    data=df, sizes=(100, 1000), legend='brief')
    
    plt.title('Throughput vs. Cost per Token (Bubble size = Model Capacity)')
    plt.xlabel('Throughput (QPS)')
    plt.ylabel('Cost ($ per Million Tokens)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

plot_response_time_throughput()
plot_rag_metrics()
plot_customer_business_metrics()
plot_cost_efficiency_bubble()


