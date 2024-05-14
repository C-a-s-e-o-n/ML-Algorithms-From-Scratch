import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance matrix
        # row = 1 sample, columns = feature vector for np
        # cov method is reversed (for some reason)
        cov = np.cov(X.T)

        # eigenvectors / eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # v[:, i]

        # sort eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1] # trick to reverse a list, so descreasing order 
        # transpose for easier ooperations 
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0:self.n_components] 

    def transform(self, X): # projection
        # project data 
        X = X - self.mean
        
        # dimension reduction
        return np.dot(X, self.components.T) # project X onto the lower dimensional components 


