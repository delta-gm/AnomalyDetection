# # PCA STEPS
# https://builtin.com/data-science/step-step-explanation-principal-component-analysis

# Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables. These combinations are done in such a way that the new variables (i.e., principal components) 
# are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. 
# So, the idea is 10-dimensional data gives you 10 principal components, but PCA tries to put maximum possible information in the first component, then maximum remaining information in the second and so on,
# principal components represent the directions of the data that explain a maximal amount of variance, that is to say, the lines that capture most information of the data. 
# The relationship between variance and information here, is that, the larger the variance carried by a line, the larger the dispersion of the data points along it, 
# and the larger the dispersion along a line, the more the information it has.

# can we guess the first principal component ? Yes, it’s approximately the line that matches the purple marks because it goes through the origin and it’s the line in which the projection of the points (red dots) 
# is the most spread out. 
# Or mathematically speaking, it’s the line that maximizes the variance (the average of the squared distances from the projected points (red dots) to the origin).

# it is eigenvectors and eigenvalues who are behind all the magic explained above, 
# because the eigenvectors of the Covariance matrix are actually the directions of the axes where there is the most variance(most information) and that we call Principal Components. 
# And eigenvalues are simply the coefficients attached to eigenvectors, which give the amount of variance carried in each Principal Component.

# By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance.

# Let’s suppose that our data set is 2-dimensional with 2 variables x,y and that the eigenvectors and eigenvalues of the covariance matrix are as follows:
# Principal Component Analysis Example

# If we rank the eigenvalues in descending order, we get λ1>λ2, which means that the eigenvector that corresponds to the first principal component (PC1) is v1 
# and the one that corresponds to the second component (PC2) isv2.

# After having the principal components, to compute the percentage of variance (information) accounted for by each component, we divide the eigenvalue of each component by the sum of eigenvalues. 
# If we apply this on the example above, we find that PC1 and PC2 carry respectively 96% and 4% of the variance of the data.

# # 4) FEATURE VECTOR
# the feature vector is simply a matrix that has as columns the eigenvectors of the components that we decide to keep. 
# This makes it the first step towards dimensionality reduction, because if we choose to keep only p eigenvectors (components) out of n, the final data set will have only p dimensions.
# # 5) RECAST THE DATA ALONG THE PRINCIPAL COMPONENTS AXES
# In this step, which is the last one, the aim is to use the feature vector formed using the eigenvectors of the covariance matrix, to reorient the data from the original axes to the ones represented by 
# the principal components (hence the name Principal Components Analysis). This can be done by multiplying the transpose of the original data set by the transpose of the feature vector.

# 1) STANDARDIZATION

df = pd.DataFrame('whatever')

def standardize(n):
    z = (n-n.mean())/n.std(ddof=1)
    return z

# 2) Covariance Matrix Computation

cov_matrix = np.cov(x_, y_)
varX = x_.var(ddof=1)
varY = y_.var(ddof=1)
print(cov_matrix, varX, varY)

# pandas: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cov.html
cov_xy = x__.cov(y__)
cov_yx = y__.cov(x__)

# 3) COMPUTE EIGENVECTORS AND EIGENVALUES OF COVARIANCE MATRIX + IDENTIFY PRINCIPLE COMPONENTS

# 4) FEATURE VECTOR: CONSTRUCT NEW SET OF AXES WITH CHOSEN EIGENVECTORS AND EIGENVALUES

# 5) RECAST DATA ON FEATURE VECTOR: FINALDATASET = FEATUREVECTOR(TRANSPOSE)*STANDARDIZEDORIGINALDATA(TRANSPOSE)