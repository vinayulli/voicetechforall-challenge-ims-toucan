"""
Example script for fine-tuning the pretrained model to your own data.

Comments in ALL CAPS are instructions
"""

import time

import torch
import wandb

from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, use_wandb, wandb_resume_id, gpu_count):
    from huggingface_hub import hf_hub_download
    from torch.utils.data import ConcatDataset

    from Modules.ToucanTTS.ToucanTTS import ToucanTTS
    from Modules.ToucanTTS.toucantts_train_loop_arbiter import train_loop
    from Utility.corpus_preparation import prepare_tts_corpus
    from Utility.storage_config import MODEL_DIR
    from Utility.storage_config import PREPROCESSING_DIR

    if gpu_id == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    assert gpu_count == 1  # distributed finetuning is not supported

    # IF YOU'RE ADDING A NEW LANGUAGE, YOU MIGHT NEED TO ADD HANDLING FOR IT IN Preprocessing/TextFrontend.py

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join(MODEL_DIR, "ToucanTTS_SYSPIN_Challenge")  # RENAME TO SOMETHING MEANINGFUL FOR YOUR DATA
    os.makedirs(save_dir, exist_ok=True)

    all_train_sets = list()  # YOU CAN HAVE MULTIPLE LANGUAGES, OR JUST ONE. JUST MAKE ONE ConcatDataset PER LANGUAGE AND ADD IT TO THE LIST.
    train_samplers = list()

    # =======================
    # =    Gujarati Data      =
    # =======================
    gujarati_datasets = list()
    gujarati_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_gujarati_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "gujarati_female"),
                                              lang="guj"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    gujarati_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_gujarati_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "gujarati_male"),
                                              lang="guj"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(gujarati_datasets))

    # ========================
    # =    English Data      =
    # ========================
    english_datasets = list()
    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_english_female(),
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "english_female"),
                                               lang="eng"))

    english_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_english_male(),
                                               corpus_dir=os.path.join(PREPROCESSING_DIR, "english_male"),
                                               lang="eng"))

    all_train_sets.append(ConcatDataset(english_datasets))

    ## Bangla Data 
    bangla_datasets = list()
    bangla_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_bengali_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "bengali_female"),
                                              lang="ben"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    bangla_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_bengali_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "bengali_male"),
                                              lang="ben"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(bangla_datasets))


    ## Bhojpuri Data 

    bhojpuri_datasets = list()
    bhojpuri_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_bhojpuri_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "bhojpuri_female"),
                                              lang="bho"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    bhojpuri_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_bhojpuri_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "bhojpuri_male"),
                                              lang="bho"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(bhojpuri_datasets))

    ## Chattisgari Data 

    chattisgari_datasets = list()
    chattisgari_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_chattisgari_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "chattisgari_female"),
                                              lang="hne"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    chattisgari_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_chattisgari_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "chattisgari_male"),
                                              lang="hne"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(chattisgari_datasets))

    ## Hindi Data 

    hindi_datasets = list()
    hindi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_hindi_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "hindi_female"),
                                              lang="hin"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    hindi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_hindi_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "hindi_male"),
                                              lang="hin"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(hindi_datasets))

    ## Kannada Data 

    kannada_datasets = list()
    kannada_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_kannada_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "kannada_female"),
                                              lang="kan"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    kannada_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_kannada_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "kannada_male"),
                                              lang="kan"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(kannada_datasets))


    ## Magahi Data 

    magahi_datasets = list()
    magahi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_magahi_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "magahi_female"),
                                              lang="mag"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    magahi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_magahi_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "magahi_male"),
                                              lang="mag"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(magahi_datasets))


    ## Maithili Data 

    maithili_datasets = list()
    maithili_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_maithili_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "maithili_female"),
                                              lang="mai"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    maithili_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_maithili_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "maithili_male"),
                                              lang="mai"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(maithili_datasets))

    ## Marathi Data 

    marathi_datasets = list()
    marathi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_marathi_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "marathi_female"),
                                              lang="mar"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    marathi_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_marathi_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "marathi_male"),
                                              lang="mar"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(marathi_datasets))

    ## Telugu Data 
    telugu_datasets = list()
    telugu_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_telugu_female(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "telugu_female"),
                                              lang="tel"))  # CHANGE THE TRANSCRIPT DICT, THE NAME OF THE CACHE DIRECTORY AND THE LANGUAGE TO YOUR NEEDS

    telugu_datasets.append(prepare_tts_corpus(transcript_dict=build_path_to_transcript_iisc_syspin_telugu_male(),
                                              corpus_dir=os.path.join(PREPROCESSING_DIR, "telugu_male"),
                                              lang="tel"))  # YOU CAN SIMPLY ADD MODE CORPORA AND DO THE SAME, BUT YOU DON'T HAVE TO, ONE IS ENOUGH

    all_train_sets.append(ConcatDataset(telugu_datasets))


































































    model = ToucanTTS()

    for train_set in all_train_sets:
        train_samplers.append(torch.utils.data.RandomSampler(train_set))

    if use_wandb:
        wandb.init(
            name=f"{__name__.split('.')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" if wandb_resume_id is None else None,
            id=wandb_resume_id,  # this is None if not specified in the command line arguments.
            resume="must" if wandb_resume_id is not None else None)

    print("Training model")
    train_loop(net=model,
               datasets=all_train_sets,
               device=device,
               save_directory=save_dir,
               batch_size=48,  # YOU MIGHT GET OUT OF MEMORY ISSUES ON SMALL GPUs, IF SO, DECREASE THIS.
               eval_lang="hin",  # THE LANGUAGE YOUR PROGRESS PLOTS WILL BE MADE IN
               warmup_steps=10000,
               lr=1e-4,  # if you have enough data (over ~1000 datapoints) you can increase this up to 1e-4 and it will still be stable, but learn quicker.
               # DOWNLOAD THESE INITIALIZATION MODELS FROM THE RELEASE PAGE OF THE GITHUB OR RUN THE DOWNLOADER SCRIPT TO GET THEM AUTOMATICALLY
               path_to_checkpoint=hf_hub_download(cache_dir=MODEL_DIR, repo_id="Flux9665/ToucanTTS", filename="ToucanTTS.pt") if resume_checkpoint is None else resume_checkpoint,
               fine_tune=True if resume_checkpoint is None and not resume else finetune,
               resume=resume,
               steps=200000,
               use_wandb=use_wandb,
               train_samplers=train_samplers,
               gpu_count=1)
    if use_wandb:
        wandb.finish()
