# 🌴 Extra Trees (Extremely Randomized Trees)

> **Prerequisites**: Random Forest | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Taking Randomness to the Extreme

### 🟢 Beginner
**Simple Explanation**: Extra Trees take Random Forests and add *another* layer of randomness. Instead of finding the absolute best way to split the data (which takes time), it picks random splits! It's faster and sometimes even better at not overfitting.

### 🟡 Intermediate
**Differences from Random Forest**: 
1. **No Bootstrap**: Uses the whole dataset for each tree.
2. **Random Splits**: Instead of calculating the optimal threshold for a feature split, it picks random thresholds.

**Comparison Chart**:
| Feature | Random Forest | Extra Trees |
| :--- | :--- | :--- |
| Bootstrapping | Yes | No (usually) |
| Split Selection | Optimal | Random Thresholds |
| Variance | Low | Extremely Low |
| Bias | Low | Slightly Higher |

### 🔴 Advanced
**Mathematics**: 
Because of the random thresholds, the computational complexity drops significantly from $O(n \log(n))$ at each node to $O(n)$. This makes Extra Trees vastly superior in high-dimensional datasets where calculating optimal splits is expensive.

---

[← Random Forest](03-Random-Forest.md) | [Back to Index](../README.md) | [Next: Voting Classifiers →](05-Voting-Classifiers.md)
