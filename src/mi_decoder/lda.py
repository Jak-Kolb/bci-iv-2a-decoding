import numpy as np


class LDA: 
    def __init__(self):
        self.classes_ = None # (K,)
        self.means_ = None # (K, d)
        self.cov_ = None # (d, d) -> d = number of features
        self.priors_ = None # (K,)
        self.W_ = None # (d, K) 
        self.intercepts_ = None # (K,)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDA":
        # X: (N, d), y: (N,)
        
        # 1. self.classes_ = np.unique(y)
        N, d = X.shape
        self.classes_, class_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        
        # 2. compute self.means_, self.priors_
        self.means_ = np.zeros((K, d))
        self.priors_ = np.zeros(K)
        for k in range(K):
            self.means_[k] = X[class_idx==k].mean(axis=0)
            self.priors_[k] = (class_idx==k).sum() / N # percentage of trials in each class

        # 3. compute pooled self.cov_ — for each class subtract its mean, accumulate (x-μ_c)(x-μ_c)^T, divide by N-K
        
        # LOOP VERSION
        # cov = np.zeros((d, d))
        # for k in range(K):
        #     R = X[class_idx==k] - self.means_[k] # (N_k, d)
        #     cov += R.T @ R # accumulate class covariance
        # self.cov_ = cov / (N - K) # pooled covariance
        
        # VECTORISED VERSION
        # we remove the signal here, isolate the noise, to determine how features fluctuate togther
        row_means = self.means_[class_idx]
        X_centered = X - row_means # average of all class trials is now 0
        self.cov_ = (X_centered.T @ X_centered) / (N - K) # (d, d) this is how every feature vector from every trial fluctuates with othe feature vectors
        
        # 4. compute self.W_ via np.linalg.solve(self.cov_, self.means_.T)
        self.W_ = np.linalg.solve(self.cov_, self.means_.T) # (d, K)
        
        
        # 5. compute self.intercepts_
        self.intercepts_ = -0.5 * np.sum(self.means_ * self.W_.T, axis=1) + np.log(self.priors_) # (K,)
        
        return self

    def predict(self, X):
        # X: (N, d) → returns (N,) class labels from self.classes_
        # scores = X @ self.W_ + self.intercepts_   # (N, K)
        # return self.classes_[scores.argmax(axis=1)
        scores = X @ self.W_ + self.intercepts_ # (N, K)
        predicted_indices = np.argmax(scores, axis=1) # (N,)
        
        return self.classes_[predicted_indices] # (N,)