############################
# The code below is a slightly adapted version of https://github.com/msesia/arc/commit/773b11bd01d22d937fdadaed6671ad5c193b6d03
#################################
import numpy as np
from sklearn.model_selection import train_test_split

def wsc(X, y, S, delta=0.1, M=1000, random_state=2020, verbose=False):
    rng = np.random.default_rng(random_state)

    def wsc_v(X, y, S, delta, v):
        n = len(y)
        cover = np.array([y[i] < S[i] for i in range(n)])
        z = np.dot(X,v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, y, S, delta, V[m])
        
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star


def wsc_unbiased(X, y, S, delta=0.1, M=1000, test_size=0.75, random_state=2020, verbose=False):
    def wsc_vab(X, y, S, v, a, b):
        n = len(y)
        cover = np.array([y[i] < S[i] for i in range(n)])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        if len(idx[0]) == 0:
            print('wsc problem')
            return 0
        coverage = np.mean(cover[idx])
        return coverage
    coverage = 0
    t = 0
    while coverage == 0 and t < 5:
        random_state=random_state + 1
        t= t+ 1
        X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=test_size,
                                                                         random_state=random_state)
        # Find adversarial parameters
        wsc_star, v_star, a_star, b_star = wsc(X_train, y_train, S_train, delta=delta, M=M, random_state=random_state, verbose=verbose)
        # Estimate coverage
        coverage = wsc_vab(X_test, y_test, S_test, v_star, a_star, b_star)
    return coverage
