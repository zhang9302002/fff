#    Copyright 2024 Flash-VStream Authors 
#
#    Licensed under the Apache License, Version 2.0 (the "License"); 
#    you may not use this file except in compliance with the License. 
#    You may obtain a copy of the License at 
#
#        http://www.apache.org/licenses/LICENSE-2.0 
#
#    Unless required by applicable law or agreed to in writing, software 
#    distributed under the License is distributed on an "AS IS" BASIS, 
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#    See the License for the specific language governing permissions and 
#    limitations under the License. 

import os
import argparse
import subprocess
import multiprocessing
import logging

def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)

# multi gpu, feature
def eval_msvd(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/MSVD-QA/video",
                    "--gt_file", "./data/eval_video/MSVD-QA/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "msvd"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1"]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "msvd"),
           "--output_dir", os.path.join(model_path, "evaluation", "msvd", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "msvd", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_msrvtt(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/MSRVTT-QA/TestVideo",
                    "--gt_file", "./data/eval_video/MSRVTT-QA/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "msrvtt"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1"]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "msrvtt"),
           "--output_dir", os.path.join(model_path, "evaluation", "msrvtt", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "msrvtt", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_actnet(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader_frame.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/ActivityNet-QA/test_frames",
                    "--gt_file", "./data/eval_video/ActivityNet-QA/test_qa.json", 
                    "--output_dir", os.path.join(model_path, "evaluation", "actnet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "actnet"),
           "--output_dir", os.path.join(model_path, "evaluation", "actnet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "actnet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_nextoe(args):  # follow msvd format, OE follow actnet
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader_frame.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/nextoe/nextoe_frames",
                    "--gt_file", "./data/eval_video/nextoe/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "nextoe"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "nextoe"),
           "--output_dir", os.path.join(model_path, "evaluation", "nextoe", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "nextoe", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_vsmovienet(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader_frame.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream/movienet_frames_renamed",
                    "--gt_file", "./data/eval_video/vstream/test_qa_movienet.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "vsmovienet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "vsmovienet"),
           "--output_dir", os.path.join(model_path, "evaluation", "vsmovienet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "vsmovienet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_vsego4d(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_loader_frame.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream/ego4d_frames",
                    "--gt_file", "./data/eval_video/vstream/test_qa_ego4d.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "vsego4d"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "vsego4d"),
           "--output_dir", os.path.join(model_path, "evaluation", "vsego4d", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "vsego4d", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_realtime_vsmovienet(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream-realtime/movienet_video_features",
                    "--gt_file", "./data/eval_video/vstream-realtime/test_qa_movienet.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsmovienet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "realtime_vsmovienet"),
           "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsmovienet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "realtime_vsmovienet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_realtime_vsego4d(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream-realtime/ego4d_video_features",
                    "--gt_file", "./data/eval_video/vstream-realtime/test_qa_ego4d.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsego4d"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "realtime_vsego4d"),
           "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsego4d", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "realtime_vsego4d", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)


def eval_egoschema(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "egoschema" + args.suffix
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_egoschema.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/EgoSchema/frames",
                    "--gt_file", "./data/eval_video/EgoSchema/test_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    ]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_egoschema_all(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "egoschema_all" + args.suffix
    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_egoschema.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/EgoSchema/frames",
                    "--gt_file", "./data/eval_video/EgoSchema/all_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    ]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_videommesub(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "videommesub_" + args.suffix
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_videomme.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/videomme/video_frames",
                    "--gt_file", "./data/eval_video/videomme/test_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    "--use_subtitle"
                    ]
            logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
            # Start subprocess and capture output
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            p = subprocess.Popen(cmd, env=my_env)
            processes.append(p)
        for idx, p in enumerate(processes):
            stdout, stderr = p.communicate()
            logging.debug(f"Subprocess {idx} stdout: {stdout}")
            if stderr:
                logging.error(f"Subprocess {idx} stderr: {stderr}")
            if p.returncode != 0:
                logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
            else:
                logging.debug(f"Subprocess {idx} completed successfully")
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_videommewo(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "videommewo_" + args.suffix
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_videomme.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/videomme/video_frames",
                    "--gt_file", "./data/eval_video/videomme/test_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    ]
            logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
            # Start subprocess and capture output
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            p = subprocess.Popen(cmd, env=my_env)
            processes.append(p)
        for idx, p in enumerate(processes):
            stdout, stderr = p.communicate()
            logging.debug(f"Subprocess {idx} stdout: {stdout}")
            if stderr:
                logging.error(f"Subprocess {idx} stderr: {stderr}")
            if p.returncode != 0:
                logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
            else:
                logging.debug(f"Subprocess {idx} completed successfully")
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)


def eval_mvbench(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "mvbench_" + args.suffix

    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_egoschema.py",
                    "--model-path", model_path,
                    "--video_dir", f"./data/eval_video/mvbench/frames",
                    "--gt_file", f"./data/eval_video/mvbench/test_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    ]
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            p = subprocess.Popen(cmd, env=my_env)
            processes.append(p)
        for idx, p in enumerate(processes):
            stdout, stderr = p.communicate()
            logging.debug(f"Subprocess {idx} stdout: {stdout}")
            if stderr:
                logging.error(f"Subprocess {idx} stderr: {stderr}")
            if p.returncode != 0:
                logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
            else:
                logging.debug(f"Subprocess {idx} completed successfully")
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

def eval_lvbench(args):  # follow msvd format, MCQ follow nextqa
    """ TODO: MCQ setting """
    model_path = args.model_path
    num_chunks = args.num_chunks
    evaluation_dir = "evaluation"
    dataset_dir = "lvbench_" + args.suffix

    if not args.test:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "flash_vstream/eval_video/model_msvd_qa_egoschema.py",
                    "--model-path", model_path,
                    "--video_dir", f"./data/eval_video/lvbench/frames",
                    "--gt_file", f"./data/eval_video/lvbench/test_qa.json",
                    "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    "--max_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    ]
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
            p = subprocess.Popen(cmd, env=my_env)
            processes.append(p)
        for idx, p in enumerate(processes):
            stdout, stderr = p.communicate()
            logging.debug(f"Subprocess {idx} stdout: {stdout}")
            if stderr:
                logging.error(f"Subprocess {idx} stderr: {stderr}")
            if p.returncode != 0:
                logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
            else:
                logging.debug(f"Subprocess {idx} completed successfully")
    cmd = ["python", "flash_vstream/eval_video/eval_activitynet_qa_egoschema.py",
           "--pred_path", os.path.join(model_path, evaluation_dir, dataset_dir),
           "--output_dir", os.path.join(model_path, evaluation_dir, dataset_dir, "results"),
           "--output_json", os.path.join(model_path, evaluation_dir, dataset_dir, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--api_key", type=str, default='')
    parser.add_argument("--api_base", type=str, default='')
    parser.add_argument("--api_type", type=str, default='')
    parser.add_argument("--api_version", type=str, default='')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--vizlen", type=int, default=0)
    parser.add_argument("--use_speech", action="store_true", default=False)
    parser.add_argument("--max_frames", type=int, default=180)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--suffix", type=str, default='')

    parser.add_argument("--T_max_frames", type=int, default=180)

    args = parser.parse_args()
    func_dic = {'msvd': eval_msvd,
                'msrvtt': eval_msrvtt,
                'actnet': eval_actnet,
                'nextoe': eval_nextoe,
                'egoschema': eval_egoschema,
                'egoschema_all': eval_egoschema_all,
                'vsmovienet': eval_vsmovienet,
                'vsego4d': eval_vsego4d,
                'realtime_vsmovienet': eval_realtime_vsmovienet,
                'realtime_vsego4d': eval_realtime_vsego4d,
                'videommesub': eval_videommesub,
                'videommewo': eval_videommewo,
                'mvbench': eval_mvbench,
                'lvbench': eval_lvbench,
                }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
