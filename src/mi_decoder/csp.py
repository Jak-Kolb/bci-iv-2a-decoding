import numpy as np
from scipy.linalg import eigh
from mi_decoder.preprocess import epoch_session
from mi_decoder.data import load_subject

class CSP: 
    
    def __init__ (self, n_components: int = 3): 
        self.n_components = n_components # number of spacial filters per class
        self.filters_ = None # shape (2*n_components, nchannels) 
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP": 
        
        # print("Shapes:", X[y == 0].transpose(0, 2, 1).shape, X[y == 1].shape)
        T = X.shape[2] # number of time points per trial
        cov_class1 = (X[y == 0] @ (X[y == 0].transpose(0, 2, 1))).mean(axis=0) / (T-1) # shape (n_trials_class1, n_channels, n_times)
        cov_class2 = (X[y == 1] @ (X[y == 1].transpose(0, 2, 1))).mean(axis=0) / (T-1) # shape (n_trials_class2, n_channels, n_times)

        _, eigenvectors = eigh(cov_class1, cov_class1 + cov_class2)
        
        # print(type(eigenvalues), eigenvalues.shape)
        # print(type(eigenvectors), eigenvectors.shape)        
        # print("Eigenvectors shape:", eigenvectors.shape)
        bottom_n = (eigenvectors[:, :self.n_components]) # smallest eigenvectors
        top_n = (eigenvectors[:, -self.n_components:]) # largest

        self.filters_ = np.concatenate((bottom_n, top_n), axis=1).T # shape (n_channels, 2*n_components)
                
        return self
    
    def transform(self, X:np.ndarray) -> np.ndarray:
        # apply spatial filters and return log variance features
        # return shape (n_trails, 2*n_components)
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted yet. Call fit() before transform().")
        
        # print(self.filters_.shape, X.shape)
        
        Z = self.filters_ @ X # (2k, C) @ (N, C, T) -> (N, 2k, T)
        
        variances = (Z ** 2).mean(axis=2) # (N, 2k)
        
        return np.log(variances / variances.sum(axis=1, keepdims=True)) # normalize and log transform
        
        
# def main(): 
    
#     data = load_subject(1)
#     X_train, y_train = epoch_session(data["0train"])
    
#     mask = (y_train == 0) | (y_train == 1)
#     Xb = X_train[mask]
#     yb = y_train[mask]
    
#     csp = CSP(n_components=3)
#     csp.fit(Xb, yb)
#     features = csp.transform(Xb)

#     print("CSP filters shape:", csp.filters_.shape)
#     print("CSP features shape:", features.shape)
#     print("CSP feature stats:", features.mean(), features.std())
    
# if __name__ == "__main__":
#     main()