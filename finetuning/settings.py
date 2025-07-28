# settings.py

# ─── Prompt Templates ───────────────────────────────────────────
PromptEOL = 'This sentence : "*sent_0*" means in one word:'
Pretended_CoT = 'After thinking step by step, this sentence : "*sent_0*" means in one word:'
Knowledge_Enhancement = (
    'The essence of a sentence is often captured by its main subjects and actions, '
    'while descriptive terms provide additional but less central details. With this in mind, '
    'this sentence : "*sent_0*" means in one word:'
)

PromptSTH = 'This sentence : "*sent_0*" means something'
PromptSUM = 'This sentence : "*sent_0*" can be summarized as'

PromptClustering = 'This sentence : "*sent_0*" belongs to the following cluster:'
QwenClustering = 'Cluster the text: *sent_0*'
noPrompt = '*sent_0*'
ClassificationPrompt1 = 'This sentence : "*sent_0*" can be classified as:'
EmotionPrompt = (
    'Classify the emotion expressed in the given Twitter "*sent_0*" ' 
    'into one of the six emotions: anger, fear, joy, love, sadness, and surprise'
)

# Choose which template your collator will use:
PROMPT_TEMPLATE = ClassificationPrompt1

# ─── Augmentation Flags ─────────────────────────────────────────
DO_DELETION   = False    # random deletion
DO_SWAP       = False    # random swap
DO_CHAR_NOISE = False    # character-level noise
ADD_EOS       = False    # append EOS token in collator

# ─── Pooling / Loss Settings ──────────────────────────────────────
POOL_MODE   = 'last'     # 'last', 'mean', or 'concat'
TEMPERATURE = 0.2        # scaling factor for cosine similarity
MARGIN      = 0.0        # margin to subtract on diagonal

# ─── Model & Data Paths ───────────────────────────────────────────
BASE_MODEL  = 'Qwen/Qwen3-0.6B' #Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B
CACHE_DIR   = '/home/user/models'
DATA_PATH   = '/home/user/models/data/augmentation_inputs/toxic_conversations_50k_qwen_positives.csv'
OUTPUT_DIR  = '/home/user/data/LLM2vec/llama_lora/test'

# ─── Training Hyperparameters ─────────────────────────────────────
BATCH_SIZE          = 120    # total batch size
MICRO_BATCH_SIZE    = 12     # per-device micro-batch size
NUM_EPOCHS          = 1
LEARNING_RATE       = 5e-5
CUTOFF_LEN          = 100    # max tokens per example
NUM_PROC            = 25     # preprocessing workers
LOAD_KBIT           = 16      # 4, 8, or 16-bit loading
WARMUP_STEPS        = 100

# ─── LoRA configuration ──────────────────────────────────────────
LORA_R             = 64
LORA_ALPHA         = 16
LORA_DROPOUT       = 0.05
LORA_TARGET_MODULES = [
    'q_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'
]

# Model dropout (hidden & attention)
DROPOUT            = 0.05

# FP16 training, logging, and saving
FP16               = True
LOG_STEPS          = 10
SAVE_STEPS         = 50
GROUP_BY_LENGTH    = False

# Logging & Misc
LOG_LEVEL          = 'INFO'
SEED               = 42
WANDB_PROJECT      = 'Finetune LORA LLM2vec'
