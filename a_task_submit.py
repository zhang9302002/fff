import argparse
import json
import os
import requests
from bytedrh2.http_client import authenticate, _get_rh2_client
from torchvision import models

# pip3 install bytedrh2 -i https://bytedpypi.byted.org/simple
# doas --krb5-username <username> rhcli init

RUN_PATH = '/mnt/bn/fasterlmmlq/workspace/Flash-VStream-Qwen2/qwen2_7b_llmseval_videomme_ws_frame540.sh'

def parse_args():
    """
    args for training
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--name', type=str, default="trail name", help='model name')
    parser.add_argument("--config", type=str,
                        default="./configs/t2i/obsidian-3b-sd15.json")
    parser.add_argument('--mode', type=str, choices=["debug", "distribute"], default="distribute",
                        help="train on single gpu or multiple gpus")
    # parser.add_argument('--data_path', type=str, default="/mnt/bn/mmdataset/dataset/")
    # parser.add_argument('--out_path', type=str, default="/mnt/bn/mmdataset/output/falcon/work_dirs/")
    parser.add_argument("--cluster", type=int, default=24, )  #
    parser.add_argument("--group", type=int, default=174)  #204:ic-cv-ViT, 174 seed, 186 seed-2
    parser.add_argument("--preempt", action='store_true')
    parser.add_argument("--gpuv", type=str, default="A100-SXM-80GB",
                        choices=["A100-SXM4-40GB", "A100-SXM-80GB", "Tesla-V100-SXM2-32GB"])
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument("--git_key", type=str, default="WK99BJqgwFbmvRyescab")
    parser.add_argument("--branch", type=str, default="unify")
    parser.add_argument("--user", type=str, default="rzw")
    args = parser.parse_args()
    return args


def submit_trial():
    json_data = {
  "resourceGroupNames": [],
  "sourceJobRunId": "e885a4e996f28703",
  "jobDefName": "compression",
  "jobDefVersionNum": 6,
  "caption": "test_2",
  "jobRunParams": {
    "envsList": {
      "PUBLIC_KEY": "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCvxTxJxf5cl0GTA1O0lrbUiSnudkwR8e63RtolMAvnSKR9rUd11IVKLx1yzvgxf1tgyxAEjPajdxARIQshlG8r+FyrIEDWPnwI75/NKG9ZU7muS01ohVATjZz6i65Ai4FKmMW23e8X4B3CBjcYh5fuZM7fS0mMBhlNyEmacY8SGkFzTKV3LV8Hd2FKcF8/AejEs2CCi7TeqGWKhdzhU5rQGlRm/aRAADkJyLBEgcm6Sv5zneziWDWsWtSRuZNHQo5Lo7QMlIy6XTpP9G7jHChAPnYxMtfv/YOtOXWWeDBAYab2W+wAja3PZZVRg/82AyfY/hfCaiLqQYb+iINzCznxywRvJpg0NPAUMeGGRah3+PEaT/uAZEDikYYkqP643pKv5CbTii1HRo8YjMXRFCcImRtbpF3o97fBG+UhDHZQnLMCm+g7uDhlq0ct6wXXMbaGeF9umgiJPVKE1L9f1jiqdel+kXgI/jTVx+N/6W67YmJRKsW1YEyV9xnQL293k0E= bytedance@Y7VMFXPY40"
    },
    "entrypointFullScript": """
bash  /mnt/bn/fasterlmmlq/workspace/run.sh 
while :; do
  sleep 300
done
    """,
    "resource": {
      "backend": "ARNOLD",
      "arnoldConfig": {
        "elasticTraining": {
          "isOpen": False
        },
        "useRobustTraining": False,
        "roles": [
          {
            "name": "worker",
            "num": 1,
            "ports": 10,
            "gpuv": "A100_SXM_80GB",
            "gpu": 8,
            "cpu": 124,
            "memory": 2011712,
          }
        ],
        "preemptible": False,
        "queuePriority": 40,
        "useTenantQuota": False,
        "batchServer": False,
        "enableRayOnArnold": False,
        "maskHosts": [],
        "queueTimeoutReminder": 0,
        "reminderInterval": 0,
        "keepMins": 0,
        "retryTimes": 0,
        "profiler": "",
        "ckptTrialId": 0,
        "bytedriveVolumes": [],
        "bytenasVolumes": [
          {
            "accessMode": "RO",
            "name": "fasterlmm",
            "roles": [
              "worker"
            ]
          },
          {
            "accessMode": "RO",
            "name": "fasterlmmlq",
            "roles": [
              "worker"
            ]
          },
          {
            "accessMode": "RO",
            "name": "longvideo",
            "roles": [
              "worker"
            ]
          }
        ],
        "hdfsVolumes": [],
        "groupIds": [
          834
        ],
        "clusterId": 17,
        "quotaPool": "default"
      }
    }
  },
  "namespace": "/topic/d4da63592ba28758",
  "tags": [
    "compression"
  ]
}
    json_str = json.dumps(json_data)
    xiaojie_tkn_cn = "ab74e9ff3122865f"
    xiaojie_tkn_va = "4c7c6f5a62a6326c"
    rzw_tkn_va = "e2cbb60dd485174c"
    authenticate(host='rh2.bytedance.net', access_token=xiaojie_tkn_cn, user_name='jinxiaojie')
    client = _get_rh2_client()
    ret = client.http_call(method='POST',
                 path='api/v1/job_run/launch',
                 data=json_str) 
    print(f'http_call, ret={ret}')

if __name__ == "__main__":
    submit_trial()
