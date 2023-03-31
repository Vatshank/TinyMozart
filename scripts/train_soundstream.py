# imports
import os
import urllib.request
import tarfile
from audiolm_pytorch import MusicLMSoundStream, SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM


BASE_PATH = "/home/ubuntu/audio-lm-stuff"
if not os.path.isdir(BASE_PATH):
  os.makedirs(BASE_PATH)
BATCH_SIZE = 2
SAVE_RESULTS_EVERY = 1000
NUM_TRAIN_STEPS = 20000
DATASETS_FOLDER = os.path.join(BASE_PATH, "datasets")
if not os.path.isdir(DATASETS_FOLDER):
  os.makedirs(DATASETS_FOLDER)

RESULTS_FOLDER = os.path.join(BASE_PATH, "results")
if not os.path.isdir(RESULTS_FOLDER):
  os.makedirs(RESULTS_FOLDER)


# TODO: move to function
# download data if it doesn't exist
# url = "https://us.openslr.org/resources/12/dev-clean.tar.gz"
# filename = os.path.join(DATASETS_FOLDER, "dev-clean")
# filename_targz = os.path.join(filename + ".tar.gz")
# if not os.path.isfile(filename_targz):
#   print("Downloading data...")
#   urllib.request.urlretrieve(url, filename_targz)
# if not os.path.isdir(filename):
#   with tarfile.open(filename_targz) as t:
#     t.extractall(filename)

folder = os.path.join(DATASETS_FOLDER, "fma_medium")


# Train soundstream for audio quality tokens
print("Initializing Soundstream model...")
# soundstream = SoundStream(
#     codebook_size = 1024,
#     rq_num_quantizers = 8,
# )

soundstream = MusicLMSoundStream()


trainer = SoundStreamTrainer(
    soundstream,
    folder = folder,
    batch_size = BATCH_SIZE,#4,
    grad_accum_every = 8,         # effective batch size of 32
    # data_max_length = 320 * 32,
    data_max_length_seconds=5,
    save_results_every = SAVE_RESULTS_EVERY,
    save_model_every = SAVE_RESULTS_EVERY,
    num_train_steps = NUM_TRAIN_STEPS,
    results_folder=RESULTS_FOLDER,
    force_clear_prev_results=True,
    is_mp3=True
).cuda()
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

print("Starting Soundstream training...")
trainer.train()

print("Done with Sounstream training.")
