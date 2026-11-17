"""Diagonal Gaussian mixtures and EM implemented with NumPy only."""
import numpy as np


def logsumexp(values, axis=-1):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)


class DiagonalGMM:
    def __init__(self, n_components=4, variance_floor=1e-3, max_iter=60,
                 tol=1e-3, n_init=2, batch_size=4096, seed=17):
        if min(n_components, max_iter, n_init, batch_size) < 1 or variance_floor <= 0 or tol < 0:
            raise ValueError("Invalid GMM hyperparameters")
        self.n_components, self.variance_floor = n_components, variance_floor
        self.max_iter, self.tol, self.n_init = max_iter, tol, n_init
        self.batch_size, self.seed = batch_size, seed

    def _initialize(self, x, rng):
        # K-means++ seeds on a bounded initialization sample; five Lloyd updates.
        sample = x[rng.choice(len(x), min(len(x), 8192), replace=False)]
        means = [sample[int(rng.integers(len(sample)))]]
        distance = ((sample - means[0]) ** 2).sum(axis=1)
        for _ in range(1, self.n_components):
            index = int(rng.choice(len(sample), p=distance / distance.sum())) if distance.sum() > 0 else int(rng.integers(len(sample)))
            means.append(sample[index])
            distance = np.minimum(distance, ((sample - means[-1]) ** 2).sum(axis=1))
        self.means_ = np.array(means)
        global_var = np.maximum(x.var(axis=0), self.variance_floor)
        self.variances_ = np.tile(global_var, (self.n_components, 1))
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        for _ in range(5):
            distances = (sample ** 2).sum(axis=1, keepdims=True) - 2 * sample @ self.means_.T + (self.means_ ** 2).sum(axis=1)
            labels = distances.argmin(axis=1)
            for k in range(self.n_components):
                group = sample[labels == k]
                if len(group):
                    self.means_[k] = group.mean(axis=0)
                    self.variances_[k] = np.maximum(group.var(axis=0), self.variance_floor)

    def _log_joint(self, x):
        inverse = 1 / self.variances_
        quadratic = (x * x) @ inverse.T - 2 * x @ (self.means_ * inverse).T + (self.means_ ** 2 * inverse).sum(axis=1)
        return (-.5 * (x.shape[1] * np.log(2 * np.pi) + np.log(self.variances_).sum(axis=1)
                       + np.maximum(quadratic, 0)) + np.log(self.weights_))

    def score_samples(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.means_.shape[1] or not np.isfinite(x).all():
            raise ValueError("Invalid GMM observations")
        return np.concatenate([logsumexp(self._log_joint(x[start:start + self.batch_size]))
                               for start in range(0, len(x), self.batch_size)]) if len(x) else np.empty(0)

    def responsibilities(self, x):
        joint = self._log_joint(np.asarray(x, dtype=np.float64))
        return np.exp(joint - logsumexp(joint)[:, None])

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < self.n_components or not np.isfinite(x).all():
            raise ValueError("Need finite observations and at least K rows")
        best, best_score = None, -np.inf
        self.initializations_ = []
        for initialization in range(self.n_init):
            rng = np.random.default_rng(self.seed + initialization * 1009)
            self._initialize(x, rng)
            history = []
            converged = False
            for iteration in range(1, self.max_iter + 1):
                nk = np.zeros(self.n_components)
                sx = np.zeros_like(self.means_)
                sx2 = np.zeros_like(self.means_)
                likelihood = 0.0
                for start in range(0, len(x), self.batch_size):
                    batch = x[start:start + self.batch_size]
                    joint = self._log_joint(batch)
                    log_density = logsumexp(joint)
                    responsibilities = np.exp(joint - log_density[:, None])
                    likelihood += float(log_density.sum())
                    nk += responsibilities.sum(axis=0)
                    sx += responsibilities.T @ batch
                    sx2 += responsibilities.T @ (batch * batch)
                average = likelihood / len(x)
                if not np.isfinite(average):
                    raise FloatingPointError("Nonfinite EM likelihood")
                history.append({"iteration": iteration, "mean_log_likelihood": average, "reinitialized": 0})
                if len(history) > 1 and not history[-2]["reinitialized"]:
                    improvement = average - history[-2]["mean_log_likelihood"]
                    if improvement < -1e-7:
                        raise FloatingPointError(f"EM likelihood decreased: {improvement}")
                    if abs(improvement) <= self.tol:
                        converged = True
                        break
                # M-step. Flooring is the constrained maximum for diagonal variances.
                alive = nk >= 1.0
                denominator = np.maximum(nk, 1e-12)[:, None]
                self.means_ = sx / denominator
                self.variances_ = np.maximum(sx2 / denominator - self.means_ ** 2, self.variance_floor)
                for k in np.flatnonzero(~alive):
                    self.means_[k] = x[int(rng.integers(len(x)))]
                    self.variances_[k] = np.maximum(x.var(axis=0), self.variance_floor)
                    nk[k] = 1.0
                    history[-1]["reinitialized"] += 1
                self.weights_ = nk / nk.sum()
            final_score = float(self.score_samples(x).mean())
            self.initializations_.append({"index": initialization, "converged": converged,
                                          "final_mean_log_likelihood": final_score, "history": history})
            if final_score > best_score:
                best_score = final_score
                best = (self.weights_.copy(), self.means_.copy(), self.variances_.copy(), initialization)
        self.weights_, self.means_, self.variances_, self.best_initialization_ = best
        self.lower_bound_ = best_score
        return self


class GMMClassifier:
    """One mixture per class; uniform priors, conditional independence of frames."""
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, sequences, labels, progress=None):
        labels = np.asarray(labels)
        if set(labels) != set(range(6)):
            raise ValueError("Training requires all six classes")
        self.models_ = []
        for label in range(6):
            x = np.concatenate([sequence for sequence, target in zip(sequences, labels) if target == label])
            parameters = {**self.parameters, "seed": self.parameters.get("seed", 17) + label * 101}
            model = DiagonalGMM(**parameters).fit(x)
            self.models_.append(model)
            if progress:
                progress(label, model)
        return self

    def decision_function(self, sequences):
        if any(not len(sequence) for sequence in sequences):
            raise ValueError("Cannot classify an empty recording")
        return np.array([[model.score_samples(sequence).mean() + np.log(1 / 6) / len(sequence)
                          for model in self.models_] for sequence in sequences])

    def save(self, path):
        np.savez(path, weights=np.stack([m.weights_ for m in self.models_]),
                 means=np.stack([m.means_ for m in self.models_]),
                 variances=np.stack([m.variances_ for m in self.models_]))

    @classmethod
    def load(cls, path):
        classifier = cls()
        with np.load(path, allow_pickle=False) as data:
            if data["weights"].shape[0] != 6:
                raise ValueError("Expected six class-conditional mixtures")
            classifier.models_ = []
            for weights, means, variances in zip(data["weights"], data["means"], data["variances"]):
                if not (np.isfinite(means).all() and np.isfinite(variances).all() and np.isfinite(weights).all()
                        and np.all(variances > 0) and np.all(weights > 0) and np.isclose(weights.sum(), 1)):
                    raise ValueError("Invalid stored mixture")
                model = DiagonalGMM(n_components=len(weights))
                model.weights_, model.means_, model.variances_ = weights, means, variances
                classifier.models_.append(model)
        return classifier
