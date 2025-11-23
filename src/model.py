from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str):
    """
    Create a token classification model using standard HuggingFace architecture.
    
    Args:
        model_name: Pre-trained model name (e.g., 'distilbert-base-uncased')
    
    Returns:
        Token classification model
    """
    # Load config first to tweak dropout
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )
    
    # Load model with the modified config
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config
    )

    return model
