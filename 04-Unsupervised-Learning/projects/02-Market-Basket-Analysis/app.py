import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

st.set_page_config(page_title="Market Basket Analysis", layout="wide")

st.title("🛒 Market Basket Analysis Dashboard")
st.markdown("Discover hidden product relationships using Association Rule Mining.")

# Generate Dummy Transaction Data
@st.cache_data
def load_data():
    np.random.seed(42)
    # 500 transactions, 10 items
    items = ['Milk', 'Bread', 'Butter', 'Eggs', 'Beer', 'Diapers', 'Cola', 'Apples', 'Chicken', 'Cheese']
    data = np.random.choice([0, 1], size=(500, len(items)), p=[0.7, 0.3])
    # Introduce some artificial correlations
    for i in range(500):
        if data[i, items.index('Beer')] == 1:
            if np.random.rand() > 0.3:
                data[i, items.index('Diapers')] = 1
        if data[i, items.index('Milk')] == 1:
            if np.random.rand() > 0.4:
                data[i, items.index('Bread')] = 1
                
    df = pd.DataFrame(data, columns=items)
    return df

df = load_data()

with st.sidebar:
    st.header("Mining Parameters")
    min_support = st.slider("Minimum Support", 0.05, 0.5, 0.1, 0.05)
    min_lift = st.slider("Minimum Lift", 0.5, 3.0, 1.0, 0.1)

# Apriori
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    
    if not rules.empty:
        # Clean up sets for display
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.subheader(f"Found {len(rules)} Rules")
        st.dataframe(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].sort_values('lift', ascending=False))
        
        # Network Graph
        st.subheader("Product Association Network")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        G = nx.from_pandas_edgelist(
            rules,
            source='antecedents_str',
            target='consequents_str',
            edge_attr='lift',
            create_using=nx.DiGraph()
        )
        
        pos = nx.spring_layout(G, k=1)
        # Node size based on degree
        d = dict(G.degree)
        node_sizes = [v * 1000 for v in d.values()]
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
                node_size=node_sizes, font_size=10, edge_color='gray', 
                arrows=True, arrowsize=20, alpha=0.8)
                
        # Draw edge labels (Lift)
        edge_labels = nx.get_edge_attributes(G, 'lift')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        
        st.pyplot(fig)
        
    else:
        st.warning("No rules found with the current thresholds. Try lowering the Lift threshold.")
else:
    st.warning("No frequent itemsets found. Try lowering the Support threshold.")
