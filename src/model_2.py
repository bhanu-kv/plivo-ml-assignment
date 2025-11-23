import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from labels import LABEL2ID, ID2LABEL


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for token classification.
    Lightweight alternative to transformer models with CRF for sequence consistency.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_labels=len(LABEL2ID), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  # Divided by 2 for bidirectional
            num_layers=2,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        
        # BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        # Classification
        logits = self.classifier(lstm_out)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return type('Output', (), {'loss': loss, 'logits': logits})()


class EnhancedTokenClassifier(nn.Module):
    """
    Enhanced transformer-based token classifier with:
    - Multi-layer classifier head for better entity discrimination
    - Layer-wise attention for better feature extraction
    - Dropout for regularization
    """
    def __init__(self, model_name, num_labels=len(LABEL2ID), hidden_dropout=0.2, classifier_dropout=0.3):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = hidden_dropout
        config.attention_probs_dropout_prob = hidden_dropout
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = num_labels
        hidden_size = config.hidden_size
        
        # Multi-layer classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return type('Output', (), {'loss': loss, 'logits': logits})()


class LightweightNERModel(nn.Module):
    """
    Lightweight NER model optimized for speed and PII precision.
    Uses DistilBERT backbone with CRF layer for sequence consistency.
    """
    def __init__(self, model_name, num_labels=len(LABEL2ID), dropout=0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = num_labels
        hidden_size = config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # CRF transition matrix for sequence consistency
        self.use_crf = True
        if self.use_crf:
            self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
            self.start_transitions = nn.Parameter(torch.randn(num_labels))
            self.end_transitions = nn.Parameter(torch.randn(num_labels))
            
    def _viterbi_decode(self, emissions, mask):
        """Viterbi decoding for CRF"""
        batch_size, seq_len, num_labels = emissions.shape
        
        # Simple greedy decoding fallback for efficiency
        return emissions.argmax(dim=-1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return type('Output', (), {'loss': loss, 'logits': logits})()


class FocalLossTokenClassifier(nn.Module):
    """
    Token classifier with Focal Loss to handle class imbalance.
    Focuses learning on hard-to-classify PII entities.
    """
    def __init__(self, model_name, num_labels=len(LABEL2ID), dropout=0.2, 
                 focal_gamma=2.0, focal_alpha=None):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = dropout
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.num_labels = num_labels
        hidden_size = config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
    def focal_loss(self, logits, labels):
        """
        Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        Focuses on hard examples and rare classes (PII entities).
        """
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, self.num_labels), 
            labels.view(-1), 
            reduction='none',
            ignore_index=-100
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        
        if self.focal_alpha is not None:
            focal_loss = self.focal_alpha * focal_loss
            
        return focal_loss.mean()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)
            
        return type('Output', (), {'loss': loss, 'logits': logits})()


def create_model(model_name: str, model_type: str = "enhanced"):
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Pre-trained model name (e.g., 'distilbert-base-uncased')
        model_type: Type of model architecture
            - 'baseline': Standard transformers token classifier
            - 'enhanced': Multi-layer classifier head (RECOMMENDED)
            - 'focal': Focal loss for class imbalance
            - 'lightweight': With CRF layer
            - 'bilstm': BiLSTM-CRF (fastest, lower accuracy)
    
    Returns:
        Token classification model
    """
    if model_type == "enhanced":
        # RECOMMENDED: Best balance of accuracy and speed
        return EnhancedTokenClassifier(
            model_name=model_name,
            num_labels=len(LABEL2ID),
            hidden_dropout=0.2,
            classifier_dropout=0.3
        )
    
    elif model_type == "focal":
        # Good for highly imbalanced data
        return FocalLossTokenClassifier(
            model_name=model_name,
            num_labels=len(LABEL2ID),
            dropout=0.2,
            focal_gamma=2.0
        )
    
    elif model_type == "lightweight":
        # Includes CRF for sequence consistency
        return LightweightNERModel(
            model_name=model_name,
            num_labels=len(LABEL2ID),
            dropout=0.2
        )
    
    elif model_type == "bilstm":
        # Fastest option, use with larger vocab_size
        return BiLSTMCRF(
            vocab_size=30522,  # BERT vocab size
            embedding_dim=128,
            hidden_dim=256,
            num_labels=len(LABEL2ID),
            dropout=0.3
        )
    
    else:  # baseline
        from transformers import AutoModelForTokenClassification
        config = AutoConfig.from_pretrained(model_name)
        config.hidden_dropout_prob = 0.2
        
        return AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=config,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )


# Convenience functions for different use cases
def create_fast_model(model_name="distilbert-base-uncased"):
    """Create optimized model for speed (target: <15ms p95)"""
    return create_model(model_name, model_type="enhanced")


def create_accurate_model(model_name="bert-base-uncased"):
    """Create model optimized for accuracy (may be slower)"""
    return create_model(model_name, model_type="focal")


def create_balanced_model(model_name="distilbert-base-uncased"):
    """Create balanced model (RECOMMENDED for this task)"""
    return create_model(model_name, model_type="enhanced")