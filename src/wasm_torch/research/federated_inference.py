"""Federated learning system for browser deployment with privacy preservation.

This module implements a novel federated learning system that enables
model training and inference across distributed browser clients while
maintaining privacy through differential privacy and secure aggregation.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..security import log_security_event, validate_path
from ..validation import validate_tensor_safe


logger = logging.getLogger(__name__)


class FederatedRole(Enum):
    """Roles in federated learning system."""
    
    COORDINATOR = "coordinator"
    CLIENT = "client"
    AGGREGATOR = "aggregator"


class AggregationStrategy(Enum):
    """Model aggregation strategies."""
    
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal" 
    SCAFFOLD = "scaffold"
    FEDNOVA = "fed_nova"
    ADAPTIVE = "adaptive_aggregation"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    
    # Core parameters
    num_clients: int = 10
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Aggregation
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    min_clients_per_round: int = 5
    client_selection_fraction: float = 0.6
    
    # Privacy
    differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Security
    secure_aggregation: bool = True
    byzantine_tolerance: bool = True
    max_byzantine_clients: int = 2
    
    # Performance
    asynchronous_updates: bool = False
    timeout_seconds: int = 300
    compression_enabled: bool = True


@dataclass
class ClientInfo:
    """Information about federated client."""
    
    client_id: str
    public_key: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
    data_samples: int = 0
    model_version: int = 0
    trust_score: float = 1.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedUpdate:
    """Model update from federated client."""
    
    client_id: str
    round_number: int
    model_weights: Dict[str, torch.Tensor]
    data_samples: int
    training_loss: float
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    computation_time: float = 0.0
    communication_time: float = 0.0
    signature: Optional[str] = None


@dataclass
class GlobalModel:
    """Global federated model state."""
    
    model: nn.Module
    version: int = 0
    round_number: int = 0
    accuracy: float = 0.0
    participants: List[str] = field(default_factory=list)
    aggregation_weights: Dict[str, float] = field(default_factory=dict)
    checkpoint_hash: Optional[str] = None


class PrivacyEngine:
    """Differential privacy engine for federated learning."""
    
    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
    def add_noise_to_gradients(
        self, 
        gradients: Dict[str, torch.Tensor],
        sensitivity: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to gradients."""
        
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            validate_tensor_safe(grad, f"gradient_{name}")
            
            # Clip gradients
            clipped_grad = self._clip_gradient(grad)
            
            # Add Gaussian noise
            noise_scale = self.noise_multiplier * sensitivity
            noise = torch.normal(0, noise_scale, size=grad.shape)
            noisy_grad = clipped_grad + noise
            
            noisy_gradients[name] = noisy_grad
            
        logger.debug(f"Added DP noise to {len(gradients)} gradients")
        return noisy_gradients
    
    def _clip_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Clip gradient norm for privacy."""
        
        grad_norm = torch.norm(gradient)
        if grad_norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / grad_norm)
            
        return gradient
    
    def compute_privacy_budget(
        self,
        epochs: int,
        batch_size: int,
        dataset_size: int,
        delta: float = 1e-5
    ) -> float:
        """Compute privacy budget (epsilon) consumed."""
        
        # Simplified privacy accounting (in practice would use more sophisticated methods)
        q = batch_size / dataset_size  # Sampling probability
        steps = epochs * (dataset_size // batch_size)
        
        # RDP accounting approximation
        epsilon = (
            q * steps * (
                np.exp(self.noise_multiplier ** (-2)) - 1
            ) / self.noise_multiplier ** 2
        )
        
        # Convert RDP to (ε, δ)-DP
        epsilon_dp = epsilon + np.log(1/delta) / (2 * steps)
        
        return float(epsilon_dp)


class SecureAggregator:
    """Secure aggregation with byzantine fault tolerance."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.client_keys: Dict[str, str] = {}
        
    def aggregate_updates(
        self,
        updates: List[FederatedUpdate],
        global_model: GlobalModel,
    ) -> Dict[str, torch.Tensor]:
        """Securely aggregate client updates."""
        
        if len(updates) < self.config.min_clients_per_round:
            raise ValueError(f"Insufficient updates: {len(updates)}")
        
        # Verify update signatures
        verified_updates = self._verify_updates(updates)
        
        # Detect and filter byzantine updates
        if self.config.byzantine_tolerance:
            verified_updates = self._filter_byzantine_updates(verified_updates)
        
        # Perform aggregation
        if self.config.aggregation_strategy == AggregationStrategy.FEDAVG:
            return self._federated_averaging(verified_updates)
        elif self.config.aggregation_strategy == AggregationStrategy.FEDPROX:
            return self._federated_proximal(verified_updates, global_model)
        elif self.config.aggregation_strategy == AggregationStrategy.ADAPTIVE:
            return self._adaptive_aggregation(verified_updates, global_model)
        else:
            return self._federated_averaging(verified_updates)
    
    def _verify_updates(self, updates: List[FederatedUpdate]) -> List[FederatedUpdate]:
        """Verify update signatures."""
        
        verified = []
        for update in updates:
            if self._verify_signature(update):
                verified.append(update)
            else:
                logger.warning(f"Invalid signature for client {update.client_id}")
                
        logger.info(f"Verified {len(verified)}/{len(updates)} updates")
        return verified
    
    def _verify_signature(self, update: FederatedUpdate) -> bool:
        """Verify update signature (simplified)."""
        
        if not self.config.secure_aggregation or not update.signature:
            return True  # Skip verification if not required
            
        # In practice would use proper cryptographic signature verification
        expected_signature = hashlib.sha256(
            f"{update.client_id}_{update.round_number}_{update.training_loss}".encode()
        ).hexdigest()[:16]
        
        return update.signature == expected_signature
    
    def _filter_byzantine_updates(
        self, 
        updates: List[FederatedUpdate]
    ) -> List[FederatedUpdate]:
        """Filter out potential byzantine updates."""
        
        if len(updates) <= self.config.max_byzantine_clients * 2:
            return updates  # Need minimum updates for detection
        
        # Compute pairwise distances between updates
        distances = self._compute_update_distances(updates)
        
        # Identify outliers (potential byzantine clients)
        outlier_indices = self._detect_outliers(distances)
        
        # Filter out outliers
        filtered_updates = [
            update for i, update in enumerate(updates)
            if i not in outlier_indices
        ]
        
        logger.info(f"Filtered {len(outlier_indices)} potential byzantine updates")
        return filtered_updates
    
    def _compute_update_distances(self, updates: List[FederatedUpdate]) -> np.ndarray:
        """Compute pairwise distances between model updates."""
        
        n_updates = len(updates)
        distances = np.zeros((n_updates, n_updates))
        
        for i in range(n_updates):
            for j in range(i + 1, n_updates):
                distance = self._compute_model_distance(
                    updates[i].model_weights,
                    updates[j].model_weights
                )
                distances[i, j] = distances[j, i] = distance
        
        return distances
    
    def _compute_model_distance(
        self,
        weights1: Dict[str, torch.Tensor],
        weights2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute distance between two model weight dictionaries."""
        
        total_distance = 0.0
        
        for name in weights1.keys():
            if name in weights2:
                w1 = weights1[name].flatten()
                w2 = weights2[name].flatten()
                distance = torch.norm(w1 - w2).item()
                total_distance += distance
        
        return total_distance
    
    def _detect_outliers(self, distances: np.ndarray, threshold: float = 2.0) -> List[int]:
        """Detect outlier updates based on distance matrix."""
        
        # Compute median distance for each update
        median_distances = np.median(distances, axis=1)
        
        # Identify outliers using modified z-score
        mad = np.median(np.abs(median_distances - np.median(median_distances)))
        modified_z_scores = 0.6745 * (median_distances - np.median(median_distances)) / mad
        
        outliers = np.where(np.abs(modified_z_scores) > threshold)[0]
        
        # Limit number of filtered updates
        if len(outliers) > self.config.max_byzantine_clients:
            # Keep only the most extreme outliers
            outlier_scores = np.abs(modified_z_scores[outliers])
            top_outliers = outliers[np.argsort(outlier_scores)[-self.config.max_byzantine_clients:]]
            return top_outliers.tolist()
        
        return outliers.tolist()
    
    def _federated_averaging(self, updates: List[FederatedUpdate]) -> Dict[str, torch.Tensor]:
        """Standard federated averaging aggregation."""
        
        total_samples = sum(update.data_samples for update in updates)
        aggregated_weights = {}
        
        # Initialize aggregated weights
        for name in updates[0].model_weights.keys():
            aggregated_weights[name] = torch.zeros_like(updates[0].model_weights[name])
        
        # Weighted averaging
        for update in updates:
            weight = update.data_samples / total_samples
            
            for name, param in update.model_weights.items():
                aggregated_weights[name] += weight * param
        
        logger.info(f"Aggregated {len(updates)} updates using FedAvg")
        return aggregated_weights
    
    def _federated_proximal(
        self,
        updates: List[FederatedUpdate],
        global_model: GlobalModel
    ) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term."""
        
        # Start with standard averaging
        aggregated = self._federated_averaging(updates)
        
        # Add proximal regularization (simplified)
        mu = 0.01  # Proximal parameter
        
        for name, param in aggregated.items():
            if name in global_model.model.state_dict():
                global_param = global_model.model.state_dict()[name]
                aggregated[name] = param + mu * global_param
        
        logger.info(f"Aggregated {len(updates)} updates using FedProx")
        return aggregated
    
    def _adaptive_aggregation(
        self,
        updates: List[FederatedUpdate], 
        global_model: GlobalModel
    ) -> Dict[str, torch.Tensor]:
        """Adaptive aggregation based on client performance."""
        
        # Compute adaptive weights based on client metrics
        adaptive_weights = []
        
        for update in updates:
            # Base weight on data samples
            base_weight = update.data_samples
            
            # Adjust based on training loss (lower loss = higher weight)
            loss_adjustment = 1.0 / (1.0 + update.training_loss)
            
            # Adjust based on validation metrics if available
            acc_adjustment = 1.0
            if 'accuracy' in update.validation_metrics:
                acc_adjustment = update.validation_metrics['accuracy']
            
            # Combine adjustments
            adaptive_weight = base_weight * loss_adjustment * acc_adjustment
            adaptive_weights.append(adaptive_weight)
        
        # Normalize weights
        total_weight = sum(adaptive_weights)
        adaptive_weights = [w / total_weight for w in adaptive_weights]
        
        # Weighted aggregation
        aggregated_weights = {}
        
        for name in updates[0].model_weights.keys():
            aggregated_weights[name] = torch.zeros_like(updates[0].model_weights[name])
        
        for weight, update in zip(adaptive_weights, updates):
            for name, param in update.model_weights.items():
                aggregated_weights[name] += weight * param
        
        logger.info(f"Aggregated {len(updates)} updates using adaptive weighting")
        return aggregated_weights


class FederatedClient:
    """Federated learning client for browser deployment."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        config: FederatedConfig,
        privacy_engine: Optional[PrivacyEngine] = None
    ):
        self.client_id = client_id
        self.model = model.to(torch.device('cpu'))  # Browser deployment
        self.config = config
        self.privacy_engine = privacy_engine or PrivacyEngine()
        
        self.round_number = 0
        self.local_data_size = 0
        self.training_metrics = {}
        
        # Generate client keypair (simplified)
        self.private_key = str(uuid.uuid4())
        self.public_key = hashlib.sha256(self.private_key.encode()).hexdigest()[:16]
    
    async def train_local_model(
        self,
        train_data: torch.utils.data.DataLoader,
        val_data: Optional[torch.utils.data.DataLoader] = None
    ) -> FederatedUpdate:
        """Train model on local data."""
        
        logger.info(f"Client {self.client_id} starting local training")
        start_time = time.time()
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_data):
                # Validate inputs
                validate_tensor_safe(data, f"training_data_batch_{batch_idx}")
                validate_tensor_safe(target, f"training_target_batch_{batch_idx}")
                
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                loss.backward()
                
                # Clip gradients for privacy
                if self.privacy_engine:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.privacy_engine.max_grad_norm
                    )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                total_samples += data.size(0)
            
            total_loss += epoch_loss
            logger.debug(f"Client {self.client_id} epoch {epoch}: loss={epoch_loss:.4f}")
        
        # Evaluate on validation data
        validation_metrics = {}
        if val_data:
            validation_metrics = await self._evaluate_model(val_data)
        
        # Get model weights
        model_weights = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Apply differential privacy
        if self.privacy_engine and self.config.differential_privacy:
            model_weights = self.privacy_engine.add_noise_to_gradients(model_weights)
        
        # Create update
        computation_time = time.time() - start_time
        
        update = FederatedUpdate(
            client_id=self.client_id,
            round_number=self.round_number,
            model_weights=model_weights,
            data_samples=total_samples,
            training_loss=total_loss / self.config.local_epochs,
            validation_metrics=validation_metrics,
            computation_time=computation_time,
            communication_time=0.0,  # Will be set during transmission
        )
        
        # Sign update
        update.signature = self._sign_update(update)
        
        logger.info(f"Client {self.client_id} completed training: "
                   f"loss={update.training_loss:.4f}, time={computation_time:.1f}s")
        
        return update
    
    async def _evaluate_model(
        self, 
        val_data: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_data:
                validate_tensor_safe(data, "validation_data")
                validate_tensor_safe(target, "validation_target")
                
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_data) if len(val_data) > 0 else float('inf')
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "samples": total,
        }
    
    def _sign_update(self, update: FederatedUpdate) -> str:
        """Sign model update for verification."""
        
        # Simplified signature (in practice would use proper cryptography)
        data_to_sign = f"{update.client_id}_{update.round_number}_{update.training_loss}"
        signature = hashlib.sha256(
            (data_to_sign + self.private_key).encode()
        ).hexdigest()[:16]
        
        return signature
    
    def update_global_model(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Update local model with global weights."""
        
        self.model.load_state_dict(global_weights)
        self.round_number += 1
        
        logger.info(f"Client {self.client_id} updated to round {self.round_number}")


class FederatedCoordinator:
    """Federated learning coordinator managing the global process."""
    
    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.global_model = GlobalModel(model=global_model)
        self.config = config
        self.aggregator = SecureAggregator(config)
        
        self.clients: Dict[str, ClientInfo] = {}
        self.round_updates: Dict[int, List[FederatedUpdate]] = {}
        self.training_history = []
        
        self.is_running = False
        
    def register_client(self, client_info: ClientInfo) -> bool:
        """Register new federated client."""
        
        if client_info.client_id in self.clients:
            logger.warning(f"Client {client_info.client_id} already registered")
            return False
        
        self.clients[client_info.client_id] = client_info
        
        log_security_event("federated_client_registered", {
            "client_id": client_info.client_id,
            "capabilities": client_info.capabilities,
        })
        
        logger.info(f"Registered client {client_info.client_id}")
        return True
    
    def select_clients_for_round(self, round_number: int) -> List[str]:
        """Select clients for current round."""
        
        # Filter available clients
        current_time = time.time()
        available_clients = [
            client_id for client_id, info in self.clients.items()
            if (current_time - info.last_seen) < self.config.timeout_seconds
        ]
        
        if len(available_clients) < self.config.min_clients_per_round:
            raise RuntimeError(f"Insufficient available clients: {len(available_clients)}")
        
        # Select fraction of clients
        num_selected = int(len(available_clients) * self.config.client_selection_fraction)
        num_selected = max(num_selected, self.config.min_clients_per_round)
        
        # Priority-based selection (trust score, data samples, etc.)
        client_scores = []
        for client_id in available_clients:
            info = self.clients[client_id]
            score = info.trust_score * (1 + info.data_samples / 1000.0)
            client_scores.append((client_id, score))
        
        # Select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        selected_clients = [client_id for client_id, _ in client_scores[:num_selected]]
        
        logger.info(f"Selected {len(selected_clients)} clients for round {round_number}")
        return selected_clients
    
    async def run_federated_round(self, round_number: int) -> Dict[str, Any]:
        """Run single federated learning round."""
        
        logger.info(f"Starting federated round {round_number}")
        start_time = time.time()
        
        # Select clients
        selected_clients = self.select_clients_for_round(round_number)
        
        # Wait for client updates (simulated)
        updates = await self._collect_client_updates(selected_clients, round_number)
        
        if len(updates) < self.config.min_clients_per_round:
            logger.warning(f"Insufficient updates received: {len(updates)}")
            return {"status": "failed", "reason": "insufficient_updates"}
        
        # Aggregate updates
        try:
            aggregated_weights = self.aggregator.aggregate_updates(updates, self.global_model)
            
            # Update global model
            self.global_model.model.load_state_dict(aggregated_weights)
            self.global_model.version += 1
            self.global_model.round_number = round_number
            self.global_model.participants = [u.client_id for u in updates]
            
            # Compute aggregation weights
            total_samples = sum(u.data_samples for u in updates)
            self.global_model.aggregation_weights = {
                u.client_id: u.data_samples / total_samples for u in updates
            }
            
            # Update client trust scores
            self._update_client_trust_scores(updates)
            
            round_time = time.time() - start_time
            
            # Record round statistics
            round_stats = {
                "round_number": round_number,
                "participants": len(updates),
                "avg_loss": np.mean([u.training_loss for u in updates]),
                "total_samples": total_samples,
                "round_time": round_time,
                "status": "success",
            }
            
            self.training_history.append(round_stats)
            
            logger.info(f"Round {round_number} completed: {len(updates)} clients, "
                       f"avg_loss={round_stats['avg_loss']:.4f}, time={round_time:.1f}s")
            
            return round_stats
            
        except Exception as e:
            logger.error(f"Round {round_number} failed: {e}")
            return {"status": "failed", "reason": str(e)}
    
    async def _collect_client_updates(
        self, 
        selected_clients: List[str], 
        round_number: int
    ) -> List[FederatedUpdate]:
        """Collect updates from selected clients (simulated)."""
        
        updates = []
        
        # In practice would communicate with actual clients
        # For simulation, create realistic updates
        for client_id in selected_clients:
            if client_id not in self.clients:
                continue
                
            client_info = self.clients[client_id]
            
            # Simulate client update
            simulated_weights = {}
            for name, param in self.global_model.model.named_parameters():
                # Add small random perturbation to simulate local training
                noise = torch.randn_like(param) * 0.01
                simulated_weights[name] = param.clone().detach() + noise
            
            update = FederatedUpdate(
                client_id=client_id,
                round_number=round_number,
                model_weights=simulated_weights,
                data_samples=client_info.data_samples,
                training_loss=0.1 + np.random.exponential(0.1),  # Realistic loss
                validation_metrics={"accuracy": 0.8 + np.random.normal(0, 0.1)},
                computation_time=5.0 + np.random.exponential(2.0),
            )
            
            updates.append(update)
            
        await asyncio.sleep(0.1)  # Simulate communication delay
        
        return updates
    
    def _update_client_trust_scores(self, updates: List[FederatedUpdate]) -> None:
        """Update client trust scores based on round performance."""
        
        # Compute metrics for trust scoring
        losses = [u.training_loss for u in updates]
        median_loss = np.median(losses)
        
        for update in updates:
            if update.client_id not in self.clients:
                continue
                
            client_info = self.clients[update.client_id]
            
            # Trust score based on loss relative to median
            loss_ratio = update.training_loss / median_loss if median_loss > 0 else 1.0
            trust_adjustment = np.clip(2.0 - loss_ratio, 0.5, 1.5)
            
            # Update trust score with exponential moving average
            alpha = 0.1
            client_info.trust_score = (
                (1 - alpha) * client_info.trust_score +
                alpha * trust_adjustment
            )
            
            # Update last seen
            client_info.last_seen = time.time()
    
    async def run_federated_training(self) -> Dict[str, Any]:
        """Run complete federated training process."""
        
        if self.is_running:
            logger.warning("Federated training already running")
            return {"status": "already_running"}
        
        self.is_running = True
        
        try:
            logger.info(f"Starting federated training for {self.config.rounds} rounds")
            
            training_results = {
                "total_rounds": self.config.rounds,
                "completed_rounds": 0,
                "round_results": [],
                "final_accuracy": 0.0,
                "total_time": 0.0,
                "status": "in_progress",
            }
            
            start_time = time.time()
            
            for round_num in range(1, self.config.rounds + 1):
                round_result = await self.run_federated_round(round_num)
                training_results["round_results"].append(round_result)
                
                if round_result["status"] == "success":
                    training_results["completed_rounds"] += 1
                else:
                    logger.warning(f"Round {round_num} failed")
                
                # Early stopping if too many failures
                failure_rate = (round_num - training_results["completed_rounds"]) / round_num
                if failure_rate > 0.3:  # More than 30% failure rate
                    logger.error("Too many round failures, stopping training")
                    training_results["status"] = "failed"
                    break
            
            training_results["total_time"] = time.time() - start_time
            
            if training_results["status"] == "in_progress":
                training_results["status"] = "completed"
                
            logger.info(f"Federated training completed: "
                       f"{training_results['completed_rounds']}/{self.config.rounds} rounds")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Federated training failed: {e}")
            return {"status": "failed", "error": str(e)}
            
        finally:
            self.is_running = False
    
    def get_global_model_state(self) -> Dict[str, Any]:
        """Get current global model state."""
        
        return {
            "version": self.global_model.version,
            "round_number": self.global_model.round_number,
            "participants": self.global_model.participants,
            "model_weights": {
                name: param.clone().detach()
                for name, param in self.global_model.model.named_parameters()
            },
            "aggregation_weights": self.global_model.aggregation_weights,
        }
    
    def export_training_report(self) -> Dict[str, Any]:
        """Export comprehensive training report."""
        
        if not self.training_history:
            return {"error": "No training history available"}
        
        successful_rounds = [r for r in self.training_history if r["status"] == "success"]
        
        report = {
            "configuration": {
                "num_clients": self.config.num_clients,
                "rounds": self.config.rounds,
                "local_epochs": self.config.local_epochs,
                "aggregation_strategy": self.config.aggregation_strategy.value,
                "differential_privacy": self.config.differential_privacy,
                "secure_aggregation": self.config.secure_aggregation,
            },
            "training_summary": {
                "total_rounds": len(self.training_history),
                "successful_rounds": len(successful_rounds),
                "success_rate": len(successful_rounds) / len(self.training_history) * 100,
                "avg_participants": np.mean([r["participants"] for r in successful_rounds]),
                "total_training_time": sum(r["round_time"] for r in successful_rounds),
            },
            "performance_metrics": {
                "final_loss": successful_rounds[-1]["avg_loss"] if successful_rounds else None,
                "loss_improvement": (
                    successful_rounds[0]["avg_loss"] - successful_rounds[-1]["avg_loss"]
                    if len(successful_rounds) >= 2 else 0
                ),
                "convergence_rounds": len(successful_rounds),
            },
            "client_statistics": {
                "total_registered": len(self.clients),
                "avg_trust_score": np.mean([c.trust_score for c in self.clients.values()]),
                "total_data_samples": sum(c.data_samples for c in self.clients.values()),
            },
            "round_history": self.training_history,
            "timestamp": time.time(),
        }
        
        return report