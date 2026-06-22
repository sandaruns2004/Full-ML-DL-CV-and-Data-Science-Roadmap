# 📜 Apriori Algorithm

> **Prerequisites**: Association Rules | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Frequent Itemsets

### 🟢 Beginner
**Simple Explanation**: The Apriori algorithm uses a smart shortcut: "If a combination of items (like Milk and Bread) is NOT frequent, then adding another item (like Milk, Bread, and Eggs) will definitely NOT be frequent." This saves the computer from checking millions of useless combinations.

### 🟡 Intermediate
**Workflow**: 
1. Set a minimum support threshold.
2. Find all individual items that meet the threshold.
3. Combine them into pairs and filter pairs meeting the threshold.
4. Combine into triplets, and so on.
5. Generate rules from the frequent itemsets using the confidence threshold.

### 🔴 Advanced
**Complexity & Alternatives**: 
Apriori requires multiple scans of the entire database, which is terrible for I/O bounds in big data. In production (Hadoop/Spark), **FP-Growth (Frequent Pattern Growth)** is heavily preferred as it only scans the database twice by building an FP-Tree in memory.

---

[← Association Rule Mining](10-Association-Rule-Mining.md) | [Back to Index](../README.md) | [Next: Anomaly Detection (Outlier Detection) →](12-Anomaly-Detection.md)
