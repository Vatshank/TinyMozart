# imports
import math
import wave
import struct
import os
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio

# For mounting to google drive - comment out if not on collab vm
# from google.colab import drive
# drive.mount('/content/gdrive')

base_path = '/content/gdrive/My Drive/colab_notebooks/tiny_mozart/audiolm_librispeech_training'
os.setwd(base_path)
# define all dataset paths, checkpoints, etc
dataset_folder = "dev-clean"
soundstream_ckpt = "results/soundstream.8.pt" # this can change depending on number of steps
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer
NUM_TRAINING_STEPS = 10000

url = "https://us.openslr.org/resources/12/dev-clean.tar.gz"
filename = "dev-clean"
filename_targz = filename + ".tar.gz"
if not os.path.isfile(filename_targz):
  urllib.request.urlretrieve(url, filename_targz)
if not os.path.isdir(filename):
  # open file
  with tarfile.open(filename_targz) as t:
    t.extractall(filename)

# Train soundstream for audio quality tokens
soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = dataset_folder, #os.path.join(base_path, dataset_folder)
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 2,
    save_model_every = 4,
    num_train_steps = NUM_TRAINING_STEPS #9
).cuda()
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()

# Train HuBert aka wav2vec for semantic tokens
if not os.path.isdir("hubert"):
  os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
  urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
  urllib.request.urlretrieve(hubert_quantizer_download, f"./{hubert_quantizer}")

wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()


trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = NUM_TRAINING_STEPS
)

trainer.train()

# Train coarse transformer
wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load(f"./{soundstream_ckpt}")

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    soundstream = soundstream,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    save_results_every = 2,
    save_model_every = 4,
    num_train_steps = NUM_TRAINING_STEPS
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()

# Fine transformer

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load(f"./{soundstream_ckpt}")

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    soundstream = soundstream,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = NUM_TRAINING_STEPS
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()

# Inference

# Everything together
audiolm = AudioLM(
    wav2vec = wav2vec,
    soundstream = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)

generated_wav = audiolm(batch_size = 1)
output_path = "out.wav"
sample_rate = 44100
torchaudio.save(output_path, generated_wav.cpu(), sample_rate)