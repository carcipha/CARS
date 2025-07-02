from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from tqdm import tqdm 
import numpy as np

def cars(X, y, n_iterations=100, ncomp = 30, verbose=True):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for feature selection from Partial Least Squares Regression (PLS-R).
    
    This function iteratively removes less relevant features based on PLS regression coefficients, using an
    exponential decay function to control the elimination rate. The optimal feature subset is determined based
    on cross-validated RMSE.

    Parameters:
    ----------
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Response variable.
    n_iterations : int, default=100
        Number of Monte Carlo iterations.
    ncomp : int, default=30
        Number of PLS components.
    verbose : bool, default=True
        If True, prints progress updates.

    Returns:
    -------
    features : numpy.ndarray
        Selected features at the optimal iteration.
    """
    np.random.seed(1)
    progress_bar = tqdm(n_iterations, desc="Progress")
    ncol = X.shape[1] # total number of features
    selected_features = list(range(ncol))  # Start with all features
    selected_features_history = []
    # exp function to determine the rate of features to remove
    p = ncol
    b=np.log(p/2)/((n_iterations-1))
    a=(p/2)**(1/(n_iterations-1))

    # Initialize variables
    rmsecv = np.zeros(n_iterations) # score per iteration
    NumLV = np.zeros(n_iterations, dtype=int) # store the number of latent variables per iteration
    Coef = np.zeros((ncol, n_iterations)) # stores the coefficients per iteration
    Nvar = np.zeros(n_iterations, dtype=int) # number of remaining variables per iteration

    for iter in range(0, n_iterations):
        X_train_subset, _, y_train_subset, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=iter) # Sampling

        Xcal = X_train_subset[:, selected_features] # subset features per iteration
        ncomp = min(ncomp, Xcal.shape[1]) # PLS components cannot be greather than the minimum number of samples

        # Fit PLS to the current feature set
        pls = PLSRegression(n_components=ncomp) 
        cv = StratifiedKFold(n_splits=5, shuffle= True, random_state=1)
        scores = cross_val_score(pls, Xcal, y_train_subset, cv=cv, scoring='neg_mean_squared_error')

        rmsecv[iter] = np.sqrt(np.abs(np.mean(scores)))

        NumLV[iter] = ncomp # get number of componets
        
        # Fit model and extract coefficients
        pls.fit(Xcal, y_train_subset)
        coef0 = np.zeros(ncol) # empty matrix with col of max features
        coef_iter = pls.coef_.flatten() # get coefficients for the current iteration

        coef0[selected_features] = coef_iter # join coeffs and subset variables in the iteration
        Coef[:, iter] = coef0 # coefficients per iteration

        # variable weigths
        # Get regression coefficients as feature importance
        weight = np.abs(pls.coef_).flatten()
        # Rank features by importance, decreasing
        ranked_features = np.argsort(-weight, kind="stable")
        
        #+++ Calculate the ratio of variables to be retained by EDF in CARS.
        ratioVariable = a * np.exp(-b * (iter+1))
        Nvar[iter] = np.count_nonzero(coef0)

        K = int(np.ceil(ncol * ratioVariable)) # number of variables to remove by force
        weight[ranked_features[K:]] = 0
        selected_features = np.array(selected_features)[ranked_features[:K]]
        selected_features_history.append(selected_features.copy())
        if verbose:

            print("iter: ",iter,"n_features: ", Nvar[iter], "rate: ", np.round(ratioVariable,3), "RMSECV: ", np.round(rmsecv[iter],3))
            progress_bar.update(1)

    # progress_bar.close()
    # Select optimal iteration
    min_error = np.min(rmsecv)
    opt_iter = np.where(rmsecv == min_error)[0][-1]  # Last occurrence
    features = selected_features_history[opt_iter]

    if verbose:
        print(f"Min RMSECV: {np.round(min_error, 3)}, Optimal iter: {opt_iter}, Retained features: {len(features)}")

    return features