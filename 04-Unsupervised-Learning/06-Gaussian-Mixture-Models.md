# ☁️ Gaussian Mixture Models (GMM)

> **Prerequisites**: Probability Distributions | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 1. Soft Clustering & EM Algorithm

### 🟢 Beginner
**Simple Explanation**: While K-Means gives a "hard" assignment (you are exactly in Cluster 1 or Cluster 2), GMM gives a "soft" assignment (you are 80% in Cluster 1, and 20% in Cluster 2). It assumes the data is made of a mixture of overlapping normal (bell-curve) distributions.

### 🟡 Intermediate
**Working Mechanism**: 
GMM uses the **Expectation-Maximization (EM)** algorithm:
- **E-Step**: Estimate the probability that each point belongs to each cluster.
- **M-Step**: Update the mean, variance, and weight of each cluster based on those probabilities.

### 🔴 Advanced
**Mathematics**: 
A GMM represents the probability of $x$ as a weighted sum of $K$ Gaussian densities:
$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$
Where $\pi_k$ are the mixing coefficients, and $\Sigma_k$ is the covariance matrix. The covariance matrix allows GMMs to model elliptical (stretched) clusters, unlike the strictly spherical clusters of K-Means.
