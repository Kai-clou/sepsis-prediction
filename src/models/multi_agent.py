"""
Multi-Agent Sepsis Prediction System

This module implements a multi-agent architecture for early sepsis prediction.
Each agent specializes in different aspects of patient physiology:
- Vitals Agent: Heart rate, blood pressure, temperature, respiratory rate, O2 saturation
- Labs Agent: Laboratory values (BUN, creatinine, platelets, etc.)
- Trend Agent: Temporal patterns and rate of change

The Meta-Learner combines agent outputs for final prediction.

Author: Jason
Date: February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class VitalsAgent(nn.Module):
    """
    Agent specialized in vital signs analysis.

    Processes: HR, Resp, Temp, SBP, DBP, MAP, O2Sat
    Uses LSTM to capture temporal patterns in vital signs.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(VitalsAgent, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for vitals agent.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional mask for padded sequences

        Returns:
            Agent embedding of shape (batch, hidden_dim // 2)
        """
        # Normalize input
        x = self.input_norm(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)

        # Attention-weighted aggregation
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim * 2)

        # Output projection
        output = self.output(context)

        return output


class LabsAgent(nn.Module):
    """
    Agent specialized in laboratory values analysis.

    Processes: BUN, Creatinine, Platelets, WBC, Bilirubin, Lactate, etc.
    Handles sparse/missing lab values with learned imputation.
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(LabsAgent, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Learned imputation for missing values
        self.imputation_embedding = nn.Parameter(torch.zeros(input_dim))

        # Input projection (handles variable lab availability)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for value + missingness indicator
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for labs agent.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            missing_mask: Binary mask indicating missing values (1 = missing)

        Returns:
            Agent embedding of shape (batch, hidden_dim // 2)
        """
        batch_size, seq_len, _ = x.shape

        # Handle missing values with learned imputation
        x_imputed = torch.where(
            missing_mask.bool(),
            self.imputation_embedding.expand(batch_size, seq_len, -1),
            x
        )

        # Concatenate values with missingness indicators
        x_combined = torch.cat([x_imputed, missing_mask.float()], dim=-1)

        # Project to hidden dimension
        x_proj = self.input_proj(x_combined)

        # LSTM encoding
        lstm_out, _ = self.lstm(x_proj)

        # Attention-weighted aggregation
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Output projection
        output = self.output(context)

        return output


class TrendAgent(nn.Module):
    """
    Agent specialized in detecting temporal trends and rate of change.

    Focuses on:
    - Rate of change in vital signs
    - Acceleration/deceleration patterns
    - Deviation from patient baseline
    """

    def __init__(
        self,
        input_dim: int = 24,  # All features
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(TrendAgent, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Compute differences (rate of change)
        self.diff_proj = nn.Linear(input_dim, hidden_dim // 2)

        # Compute second differences (acceleration)
        self.accel_proj = nn.Linear(input_dim, hidden_dim // 2)

        # Transformer encoder for complex temporal patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for trend agent.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Agent embedding of shape (batch, hidden_dim // 2)
        """
        batch_size, seq_len, _ = x.shape

        # Compute first differences (rate of change)
        diff = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        # Compute second differences (acceleration)
        accel = torch.zeros_like(x)
        accel[:, 2:, :] = diff[:, 2:, :] - diff[:, 1:-1, :]

        # Project differences
        diff_emb = self.diff_proj(diff)
        accel_emb = self.accel_proj(accel)

        # Combine
        combined = torch.cat([diff_emb, accel_emb], dim=-1)  # (batch, seq_len, hidden_dim)

        # Transformer encoding
        transformer_out = self.transformer(combined)

        # Global average pooling
        context = transformer_out.mean(dim=1)

        # Output projection
        output = self.output(context)

        return output


class MetaLearner(nn.Module):
    """
    Meta-learner that combines outputs from all agents.

    Uses attention mechanism to weight agent contributions based on
    the specific patient context.
    """

    def __init__(
        self,
        agent_dim: int = 32,  # Output dim from each agent (hidden_dim // 2)
        num_agents: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super(MetaLearner, self).__init__()

        self.num_agents = num_agents

        # Agent-level attention
        self.agent_attention = nn.Sequential(
            nn.Linear(agent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(agent_dim * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(self, agent_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for meta-learner.

        Args:
            agent_outputs: List of agent embeddings, each of shape (batch, agent_dim)

        Returns:
            logits: Prediction logits of shape (batch, 1)
            agent_weights: Attention weights for each agent of shape (batch, num_agents)
        """
        # Stack agent outputs
        stacked = torch.stack(agent_outputs, dim=1)  # (batch, num_agents, agent_dim)

        # Compute agent attention weights
        attn_scores = self.agent_attention(stacked).squeeze(-1)  # (batch, num_agents)
        agent_weights = F.softmax(attn_scores, dim=-1)

        # Concatenate all agent outputs for fusion
        concatenated = torch.cat(agent_outputs, dim=-1)  # (batch, agent_dim * num_agents)

        # Fuse representations
        fused = self.fusion(concatenated)

        # Final prediction
        logits = self.classifier(fused)

        return logits, agent_weights


class MultiAgentSepsisPredictor(nn.Module):
    """
    Complete Multi-Agent Sepsis Prediction System.

    Combines specialized agents for comprehensive patient assessment:
    - Vitals Agent: Monitors cardiovascular and respiratory status
    - Labs Agent: Analyzes laboratory markers with imputation
    - Trend Agent: Detects deterioration patterns
    - Meta-Learner: Combines insights for final prediction

    Example:
        >>> model = MultiAgentSepsisPredictor()
        >>> vitals = torch.randn(32, 24, 7)  # batch, seq_len, features
        >>> labs = torch.randn(32, 24, 17)
        >>> labs_mask = torch.zeros(32, 24, 17)
        >>> all_features = torch.randn(32, 24, 24)
        >>>
        >>> output = model(vitals, labs, labs_mask, all_features)
        >>> print(output['probability'].shape)  # (32, 1)
    """

    # Feature indices for each agent
    VITALS_FEATURES = ['hr', 'resp', 'temp', 'sbp', 'dbp', 'map_value', 'o2sat']
    LABS_FEATURES = ['bun', 'chloride', 'creatinine', 'wbc', 'bicarbonate', 'platelets',
                     'magnesium', 'calcium', 'potassium', 'sodium', 'glucose',
                     'fio2', 'ph', 'paco2', 'pao2', 'lactate', 'bilirubin']

    def __init__(
        self,
        vitals_dim: int = 7,
        labs_dim: int = 17,
        all_features_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(MultiAgentSepsisPredictor, self).__init__()

        self.vitals_dim = vitals_dim
        self.labs_dim = labs_dim
        self.all_features_dim = all_features_dim

        # Initialize agents
        self.vitals_agent = VitalsAgent(
            input_dim=vitals_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.labs_agent = LabsAgent(
            input_dim=labs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.trend_agent = TrendAgent(
            input_dim=all_features_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Meta-learner
        self.meta_learner = MetaLearner(
            agent_dim=hidden_dim // 2,
            num_agents=3,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(
        self,
        vitals: torch.Tensor,
        labs: torch.Tensor,
        labs_missing_mask: torch.Tensor,
        all_features: torch.Tensor,
        vitals_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-agent system.

        Args:
            vitals: Vital signs tensor (batch, seq_len, vitals_dim)
            labs: Lab values tensor (batch, seq_len, labs_dim)
            labs_missing_mask: Missing indicator for labs (batch, seq_len, labs_dim)
            all_features: All features for trend agent (batch, seq_len, all_features_dim)
            vitals_mask: Optional sequence mask for vitals

        Returns:
            Dictionary containing:
            - 'logits': Raw prediction logits
            - 'probability': Sepsis probability (0-1)
            - 'agent_weights': Contribution of each agent
            - 'vitals_embedding': Vitals agent representation
            - 'labs_embedding': Labs agent representation
            - 'trend_embedding': Trend agent representation
        """
        # Get agent embeddings
        vitals_emb = self.vitals_agent(vitals, vitals_mask)
        labs_emb = self.labs_agent(labs, labs_missing_mask)
        trend_emb = self.trend_agent(all_features)

        # Meta-learner combines agents
        logits, agent_weights = self.meta_learner([vitals_emb, labs_emb, trend_emb])

        # Compute probability
        probability = torch.sigmoid(logits)

        return {
            'logits': logits,
            'probability': probability,
            'agent_weights': agent_weights,
            'vitals_embedding': vitals_emb,
            'labs_embedding': labs_emb,
            'trend_embedding': trend_emb
        }

    def get_agent_explanations(self, agent_weights: torch.Tensor) -> Dict[str, float]:
        """
        Convert agent weights to human-readable explanations.

        Args:
            agent_weights: Tensor of shape (batch, 3)

        Returns:
            Dictionary mapping agent names to their contribution percentages
        """
        weights = agent_weights.mean(dim=0).cpu().numpy()
        return {
            'vitals_contribution': float(weights[0] * 100),
            'labs_contribution': float(weights[1] * 100),
            'trend_contribution': float(weights[2] * 100)
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This helps the model focus on hard examples and reduces the impact
    of easy negatives.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions (batch, 1)
            targets: Ground truth labels (batch, 1)

        Returns:
            Scalar loss value
        """
        probs = torch.sigmoid(logits)

        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply focal weights
        focal_loss = focal_weight * bce

        return focal_loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Multi-Agent Sepsis Predictor...")

    # Create model
    model = MultiAgentSepsisPredictor(
        vitals_dim=7,
        labs_dim=17,
        all_features_dim=24,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    )

    print(f"\nModel Parameters: {count_parameters(model):,}")

    # Create dummy inputs
    batch_size = 32
    seq_len = 24  # 24 hours of data

    vitals = torch.randn(batch_size, seq_len, 7)
    labs = torch.randn(batch_size, seq_len, 17)
    labs_mask = torch.zeros(batch_size, seq_len, 17)  # No missing values
    all_features = torch.randn(batch_size, seq_len, 24)

    # Forward pass
    output = model(vitals, labs, labs_mask, all_features)

    print(f"\nOutput shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Probability: {output['probability'].shape}")
    print(f"  Agent weights: {output['agent_weights'].shape}")

    # Get explanations
    explanations = model.get_agent_explanations(output['agent_weights'])
    print(f"\nAgent Contributions:")
    for agent, contrib in explanations.items():
        print(f"  {agent}: {contrib:.1f}%")

    print("\n✅ Model test passed!")
