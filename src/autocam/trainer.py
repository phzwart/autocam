"""Dynamic self-supervised learning trainer for Autocam conferences."""

import random
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import torch


class ModelInterface(ABC):
    """Abstract interface for models that can be trained."""

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Get the expected input shape."""
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        """Get the output shape."""
        pass

    @abstractmethod
    def train(self):
        """Set model to training mode."""
        pass

    @abstractmethod
    def eval(self):
        """Set model to evaluation mode."""
        pass


class DynamicSelfSupervisedTrainer:
    """Dynamic trainer that handles different training mandates."""

    def __init__(self, models: Dict[str, ModelInterface]):
        """Initialize trainer with models.

        Args:
            models: Dictionary mapping model names to model instances
        """
        self.models = models
        self.training_history = {}

    def train_batch_one_vs_one(
        self,
        batch,
        student_participants: List[str],
        target_participants: List[str],
        loss_fn,
        optimizers,
    ):
        """Train models in one vs one configuration.

        Args:
            batch: Input data batch
            student_participants: List of student model names
            target_participants: List of target model names
            loss_fn: Loss function
            optimizers: Dictionary of optimizers

        Returns:
            Dictionary of training results
        """
        results = {}

        for i, student_name in enumerate(student_participants):
            if i < len(target_participants):
                target_name = target_participants[i]
            else:
                # If no corresponding target, pick random one
                target_name = random.choice(target_participants)  # nosec

            if student_name in self.models and target_name in self.models:
                student_model = self.models[student_name]
                target_model = self.models[target_name]

                # Forward pass
                student_output = student_model.forward(batch)
                with torch.no_grad():
                    target_output = target_model.forward(batch)

                # Compute loss
                loss = loss_fn(student_output, target_output)

                # Backward pass
                if student_name in optimizers:
                    optimizers[student_name].zero_grad()
                    loss.backward()
                    optimizers[student_name].step()

                results[f"{student_name}_vs_{target_name}"] = {
                    "loss": loss.item(),
                    "student_output_shape": student_output.shape,
                    "target_output_shape": target_output.shape,
                }

        return results

    def train_batch_one_vs_random_mean(
        self,
        batch,
        student_participants: List[str],
        random_mean_count: int,
        loss_fn,
        optimizers,
    ):
        """Train models against mean of random model outputs.

        Args:
            batch: Input data batch
            student_participants: List of student model names
            random_mean_count: Number of models to use in mean
            loss_fn: Loss function
            optimizers: Dictionary of optimizers

        Returns:
            Dictionary of training results
        """
        results = {}
        available_models = list(self.models.keys())

        for student_name in student_participants:
            if (
                student_name in self.models
                and len(available_models) >= random_mean_count
            ):
                # Choose random models for mean
                k = random.randint(
                    2, min(random_mean_count, len(available_models))
                )  # nosec
                chosen_models = random.sample(available_models, k)  # nosec

                # Compute mean of random model outputs
                mean_output = None
                for model_name in chosen_models:
                    if model_name != student_name:
                        model_output = self.models[model_name].forward(batch)
                        if mean_output is None:
                            mean_output = model_output
                        else:
                            mean_output = (mean_output + model_output) / 2

                if mean_output is not None:
                    student_output = self.models[student_name].forward(batch)
                    loss = loss_fn(student_output, mean_output)

                    if student_name in optimizers:
                        optimizers[student_name].zero_grad()
                        loss.backward()
                        optimizers[student_name].step()

                    results[f"{student_name}_vs_random_mean"] = {
                        "loss": loss.item(),
                        "mean_models": chosen_models,
                        "output_shape": student_output.shape,
                    }

        return results

    def train_batch_one_vs_fixed(
        self,
        batch,
        student_participants: List[str],
        fixed_target: str,
        loss_fn,
        optimizers,
    ):
        """Train models against a fixed target model.

        Args:
            batch: Input data batch
            student_participants: List of student model names
            fixed_target: Name of fixed target model
            loss_fn: Loss function
            optimizers: Dictionary of optimizers

        Returns:
            Dictionary of training results
        """
        results = {}

        if fixed_target not in self.models:
            return results

        target_model = self.models[fixed_target]

        for student_name in student_participants:
            if student_name in self.models and student_name != fixed_target:
                student_model = self.models[student_name]

                # Forward pass
                student_output = student_model.forward(batch)
                with torch.no_grad():
                    target_output = target_model.forward(batch)

                # Compute loss
                loss = loss_fn(student_output, target_output)

                # Backward pass
                if student_name in optimizers:
                    optimizers[student_name].zero_grad()
                    loss.backward()
                    optimizers[student_name].step()

                results[f"{student_name}_vs_{fixed_target}"] = {
                    "loss": loss.item(),
                    "student_output_shape": student_output.shape,
                    "target_output_shape": target_output.shape,
                }

        return results

    def train_batch_random_pairs(
        self, batch, participants: List[str], loss_fn, optimizers
    ):
        """Train models in random student-target pairs.

        Args:
            batch: Input data batch
            participants: List of participant names
            loss_fn: Loss function
            optimizers: Dictionary of optimizers

        Returns:
            Dictionary of training results
        """
        results = {}
        available_participants = [p for p in participants if p in self.models]

        if len(available_participants) < 2:
            return results

        # Randomly pair participants
        random.shuffle(available_participants)
        pairs = []
        for i in range(0, len(available_participants) - 1, 2):
            pairs.append((available_participants[i], available_participants[i + 1]))

        for student_name, target_name in pairs:
            student_model = self.models[student_name]
            target_model = self.models[target_name]

            # Forward pass
            student_output = student_model.forward(batch)
            with torch.no_grad():
                target_output = target_model.forward(batch)

            # Compute loss
            loss = loss_fn(student_output, target_output)

            # Backward pass
            if student_name in optimizers:
                optimizers[student_name].zero_grad()
                loss.backward()
                optimizers[student_name].step()

            results[f"{student_name}_vs_{target_name}"] = {
                "loss": loss.item(),
                "student_output_shape": student_output.shape,
                "target_output_shape": target_output.shape,
            }

        return results

    def train_batch_barycentric_targets(
        self,
        batch,
        student_participants: List[str],
        min_models: int,
        max_models: int,
        loss_fn,
        optimizers,
    ):
        """Train models against barycentric combinations of other models.

        Args:
            batch: Input data batch
            student_participants: List of student model names
            min_models: Minimum number of models in combination
            max_models: Maximum number of models in combination
            loss_fn: Loss function
            optimizers: Dictionary of optimizers

        Returns:
            Dictionary of training results
        """
        results = {}
        available_models = list(self.models.keys())

        for student_name in student_participants:
            if student_name in self.models and len(available_models) >= min_models:
                # Choose random number of models for barycentric combination
                k = random.randint(
                    min_models, min(max_models, len(available_models))
                )  # nosec
                chosen_models = random.sample(available_models, k)  # nosec

                # Compute barycentric combination
                barycentric_output = None
                weights = np.random.dirichlet(np.ones(k))

                for i, model_name in enumerate(chosen_models):
                    if model_name != student_name:
                        model_output = self.models[model_name].forward(batch)
                        if barycentric_output is None:
                            barycentric_output = weights[i] * model_output
                        else:
                            barycentric_output += weights[i] * model_output

                if barycentric_output is not None:
                    student_output = self.models[student_name].forward(batch)
                    loss = loss_fn(student_output, barycentric_output)

                    if student_name in optimizers:
                        optimizers[student_name].zero_grad()
                        loss.backward()
                        optimizers[student_name].step()

                    results[f"{student_name}_vs_barycentric"] = {
                        "loss": loss.item(),
                        "barycentric_models": chosen_models,
                        "weights": weights.tolist(),
                        "output_shape": student_output.shape,
                    }

        return results

    def get_training_history(self) -> Dict[str, Any]:
        """Get training history.

        Returns:
            Dictionary containing training history
        """
        return self.training_history

    def reset_training_history(self):
        """Reset training history."""
        self.training_history = {}


def normalize_features(x):
    """Normalize features using L2 normalization.

    Args:
        x: Input features

    Returns:
        Normalized features
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = np.where(x_norm == 0, 1, x_norm)
    return x / x_norm


def compute_similarity_matrix(features):
    """Compute similarity matrix between features.

    Args:
        features: Feature matrix

    Returns:
        Similarity matrix
    """
    normalized_features = normalize_features(features)
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    return similarity_matrix


def extract_features(models, dataloader):
    """Extract features from models using dataloader.

    Args:
        models: Dictionary of models
        dataloader: Data loader

    Returns:
        Dictionary mapping model names to feature arrays
    """
    features = {}

    for model_name, model in models.items():
        model.eval()
        all_features = []

        try:
            for batch in dataloader:
                with torch.no_grad():
                    batch_features = model.forward(batch)
                    all_features.append(batch_features.cpu().numpy())

            features[model_name] = np.concatenate(all_features, axis=0)
        except Exception as e:
            print(f"Error extracting features from {model_name}: {e}")
            features[model_name] = None

    return features


def compute_convergence_metrics(features_dict):
    """Compute convergence metrics between model features.

    Args:
        features_dict: Dictionary mapping model names to feature arrays

    Returns:
        Dictionary of convergence metrics
    """
    metrics = {}
    model_names = list(features_dict.keys())

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Avoid duplicate pairs
                features1 = features_dict[model1]
                features2 = features_dict[model2]

                if features1 is not None and features2 is not None:
                    # Compute similarity
                    similarity = compute_similarity_matrix(
                        np.concatenate([features1, features2], axis=0)
                    )

                    # Compute convergence metric
                    convergence_score = np.mean(similarity)

                    metrics[f"{model1}_vs_{model2}"] = {
                        "convergence_score": convergence_score,
                        "feature_similarity": similarity.tolist(),
                    }

    return metrics


def stack_latent_vectors(features_dict):
    """Stack latent vectors from all models.

    Args:
        features_dict: Dictionary mapping model names to feature arrays

    Returns:
        Stacked feature matrix
    """
    valid_features = []
    model_names = []

    for model_name, features in features_dict.items():
        if features is not None:
            valid_features.append(features)
            model_names.append(model_name)

    if not valid_features:
        return None, []

    stacked_features = np.concatenate(valid_features, axis=1)
    return stacked_features, model_names


def perform_svd_on_stacked_features(stacked_features, n_components=None):
    """Perform SVD on stacked features.

    Args:
        stacked_features: Stacked feature matrix
        n_components: Number of components to keep

    Returns:
        Dictionary containing SVD results
    """
    if stacked_features is None:
        return None

    try:
        from sklearn.decomposition import TruncatedSVD

        if n_components is None:
            n_components = min(stacked_features.shape[1], 100)

        svd = TruncatedSVD(n_components=n_components)
        svd_features = svd.fit_transform(stacked_features)

        return {
            "svd_features": svd_features,
            "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
            "singular_values": svd.singular_values_.tolist(),
            "n_components": n_components,
        }
    except ImportError:
        print("sklearn not available, using numpy SVD")
        u, s, vt = np.linalg.svd(stacked_features, full_matrices=False)

        if n_components is not None:
            u = u[:, :n_components]
            s = s[:n_components]
            vt = vt[:n_components, :]

        return {
            "svd_features": u,
            "singular_values": s.tolist(),
            "Vt": vt.tolist(),
            "n_components": n_components or len(s),
        }
