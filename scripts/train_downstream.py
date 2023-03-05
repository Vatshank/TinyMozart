import os
import urllib.request
from audiolm_pytorch import SoundStream, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerTrainer, FineTransformer, FineTransformerTrainer, AudioLM
import torchaudio

BATCH_SIZE = 4
SAVE_RESULTS_EVERY = 2
NUM_TRAIN_STEPS = 20
SS_CHECKPOINT = 18

BASE_PATH = "/home/ubuntu/audio-lm-stuff"
DATASETS_FOLDER = os.path.join(BASE_PATH, "datasets")
RESULTS_FOLDER = os.path.join(BASE_PATH, "results")

# define all dataset paths, checkpoints, etc
soundstream_ckpt = os.path.join(RESULTS_FOLDER, f"soundstream.{SS_CHECKPOINT}.pt") # this can change depending on number of steps
print("Loading Soundstream from checkpoint...")
soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)
soundstream.load(soundstream_ckpt)
print("Finished loading Soundstream.")

# TODO: move to function
hubert_folder = os.path.join(DATASETS_FOLDER, "hubert")
hubert_ckpt = os.path.join(hubert_folder, "hubert_base_ls960.pt")
hubert_quantizer = os.path.join(hubert_folder, "hubert_base_ls960_L9_km500.bin") # listed in row "HuBERT Base (~95M params)", column Quantizer


# Train HuBert aka wav2vec for semantic tokens
if not os.path.isdir(hubert_folder):
  os.makedirs(hubert_folder)
if not os.path.isfile(hubert_ckpt):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
  urllib.request.urlretrieve(hubert_ckpt_download, hubert_ckpt)
if not os.path.isfile(hubert_quantizer):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin"
  urllib.request.urlretrieve(hubert_quantizer_download, hubert_quantizer)

print("Initializing wav2vec with Hubert checkpoints...")
wav2vec = HubertWithKmeans(
    checkpoint_path = hubert_ckpt,
    kmeans_path = hubert_quantizer,
)

print("Initializing semantic transformer...")
semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()


print("Training semantic transformer...")
trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = DATASETS_FOLDER,
    batch_size = BATCH_SIZE,
    data_max_length = 320 * 32,
    num_train_steps = NUM_TRAIN_STEPS,
    results_folder=RESULTS_FOLDER,
    force_clear_prev_results=False
)

trainer.train()
print("Finished training semantic transformer.")

# Train coarse transformer
print("Initializing coarse transformer...")
coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

print("Training coarse transformer...")
trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    soundstream = soundstream,
    wav2vec = wav2vec,
    folder = DATASETS_FOLDER,
    batch_size = BATCH_SIZE,
    data_max_length = 320 * 32,
    save_results_every = SAVE_RESULTS_EVERY,
    save_model_every = SAVE_RESULTS_EVERY,
    num_train_steps = NUM_TRAIN_STEPS,
    results_folder=RESULTS_FOLDER,
    force_clear_prev_results=False
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()
print("Finished training coarse transformer.")


# Fine transformer
print("Initializing fine transformer...")
fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

print("Training fine transformer...")
trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    soundstream = soundstream,
    folder = DATASETS_FOLDER,
    batch_size = BATCH_SIZE,
    data_max_length = 320 * 32,
    num_train_steps = NUM_TRAIN_STEPS,
    results_folder=RESULTS_FOLDER,
    force_clear_prev_results=False
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()
print("Finished training coarse transformer.")


# Inference
# Everything together
print("Initializing AudioLM...")
audiolm = AudioLM(
    wav2vec = wav2vec,
    soundstream = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)

print("Generating output...")
generated_wav = audiolm(batch_size = 1)
output_path = os.path.join(RESULTS_FOLDER, "out.wav")
sample_rate = 44100
torchaudio.save(output_path, generated_wav.cpu(), sample_rate)

print("Fin.")
