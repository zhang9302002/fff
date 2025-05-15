import json
import os

json_files = [
    "chatunivi/0509_chatunivi_vs_ego_pred.json",
    "chatunivi/0509_chatunivi_vs_movie_pred.json",
    "llamavid/LLaMAVID_vs_ego_pred_addlong.json",
    "llamavid/LLaMAVID_vs_movie_pred.json",
    "moviechat/MovieChat_vs_ego_pred.json",
    "moviechat/MovieChat_vs_movie_pred.json",
    "videochatgpt/VideoChatGPT_vs_ego_pred_short.json",
    "videochatgpt/VideoChatGPT_vs_movie_pred_short.json"
]

src_base = 'output/old_model'
tgt_base = 'output/old_model_organized'

