import pickle 
from pathlib import Path 
import fire 
import shutil 
import numpy as np
import os

import paddle
import time 
from params.configs import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from data.preprocess import merge_second_batch
import core 
import models 
import libs 
from libs.tools.progress_bar import ProgressBar
import data.kitti_common as kitti
from libs.tools.eval import get_official_eval_result, get_coco_eval_result



def train(cfg_file = None,
          model_dir = None,
          result_path=None,
          create_folder=False,
          display_step=2,
          summary_step=5,
          pickle_result=False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    shutil.copyfile(cfg_file, str(model_dir / config_file_bkp))

    config = cfg_from_yaml_file(cfg_file, cfg)
    input_cfg = config.TRAIN_INPUT_READER
    eval_input_cfg = config.EVAL_INPUT_READER
    model_cfg = config.MODEL
    train_cfg = config.TRAIN_CONFIG
    class_names = config.CLASS_NAMES 
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = core.build_voxel_generator(config.VOXEL_GENERATOR)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = core.build_box_coder(config.BOX_CODER)
    target_assigner_cfg = config.TARGET_ASSIGNER
    target_assigner = core.build_target_assigner(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.POST_PROCESSING.post_center_limit_range
    net = models.build_network(model_cfg, voxel_generator, target_assigner)
    print("num_trainable parameters:", len(list(net.parameters())))
    # for n, p in net.named_parameters():
    #     print(n, p.shape)

    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    libs.tools.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.OPTIMIZER
    if train_cfg.ENABLE_MIXED_PRECISION:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = core.build_optimizer(optimizer_cfg, net.parameters())
    if train_cfg.ENABLE_MIXED_PRECISION:
        loss_scale = train_cfg.LOSS_SCALE_FACTOR
        mixed_optimizer = libs.tools.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
   # must restore optimizer AFTER using MixedPrecisionWrapper
    libs.tools.try_restore_latest_checkpoints(model_dir,
                                              [mixed_optimizer])
    # lr_scheduler = core.build_lr_schedules(optimizer_cfg, optimizer, gstep)
    if train_cfg.ENABLE_MIXED_PRECISION:
        float_dtype = paddle.float16
    else:
        float_dtype = paddle.float32
    ######################
    # PREPARE INPUT
    ######################
    dataset = core.build_input_reader(input_cfg,
                                      model_cfg,
                                      training= True,
                                      voxel_generator=voxel_generator,
                                      target_assigner=target_assigner)
    eval_dataset = core.build_input_reader(input_cfg,
                                           model_cfg,
                                           training= False,
                                           voxel_generator=voxel_generator,
                                           target_assigner=target_assigner)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(),dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
    # print("++++++++++++++++++++++++++++++++++++START LOADER++++++++++++++++++++++++++++++++++++++++++++++++")
    dataloader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=input_cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=input_cfg.NUM_WORKERS,
        collate_fn=merge_second_batch)
    eval_dataloader = paddle.io.DataLoader(
        dataset=eval_dataset,
        batch_size = eval_input_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=eval_input_cfg.NUM_WORKERS,
        collate_fn=merge_second_batch)
    data_iter = iter(dataloader)

    ######################
    # TRAINING
    ######################
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    # logf.write(proto_str)
    logf.write("\n")

    total_step_elapsed = 0
    remain_steps = train_cfg.STEPS - net.get_global_step()
    t = time.time()
    pd_start_time = t
    #total_loop = train_cfg.STEPS // train_cfg.STEPS_PER_EVAL + 1
    total_loop = remain_steps // train_cfg.STEPS_PER_EVAL + 1
    clear_metrics_every_epoch = train_cfg.CLEAR_METRICS_EVERY_EPOCH

    if train_cfg.STEPS % train_cfg.STEPS_PER_EVAL == 0:
        total_loop -= 1
    mixed_optimizer.clear_grad()
    # print("++++++++++++++++++++++++++++++++++++TRAIN PREPARE++++++++++++++++++++++++++++++++++++++++++++++++")
    try:
        # print("++++++++++++++++++++++++++++++++++++START TRAIN++++++++++++++++++++++++++++++++++++++++++++++++")
        for _ in range(total_loop):
            # print("++++++++++++++++++++++++++++++++++++START LOOP++++++++++++++++++++++++++++++++++++++++++++++++")
            if total_step_elapsed + train_cfg.STEPS_PER_EVAL > train_cfg.STEPS:
                steps = train_cfg.STEPS % train_cfg.STEPS_PER_EVAL
            else:
                steps = train_cfg.STEPS_PER_EVAL

            for step in range(steps):
                # print("++++++++++++++++++++++++++++++++++++START STEP++++++++++++++++++++++++++++++++++++++++++++++++")
                optimizer.step()
                # print("++++++++++++++++++++++++++++++++++++OVER STEP++++++++++++++++++++++++++++++++++++++++++++++++")
                try:
                    # print("++++++++++++++++++++++++++++++++++++START EXAMPLE++++++++++++++++++++++++++++++++++++++++++++++++")
                    example = next(data_iter)
                    # print("++++++++++++++++++++++++++++++++++++OVER EXAMPLE++++++++++++++++++++++++++++++++++++++++++++++++")
                except StopIteration:
                    print("end epoch")
                    if clear_metrics_every_epoch:
                        net.clear_metrics()
                    data_iter = iter(dataloader)
                    example = next(data_iter)

                # print("++++++++++++++++++++++++++++++++++++START EXAMPLE_PADDLE++++++++++++++++++++++++++++++++++++++++++++++++")
                example_paddle = example_convert_to_paddle(example, float_dtype)
                # print("++++++++++++++++++++++++++++++++++++OVER EXAMPLE_PADDLE++++++++++++++++++++++++++++++++++++++++++++++++")
                batch_size = example["anchors"].shape[0]

                # print("++++++++++++++++++++++++++++++++++++START NET_TRAIN++++++++++++++++++++++++++++++++++++++++++++++++")
                ret_dict = net(example_paddle)
                # print("++++++++++++++++++++++++++++++++++++OVER NET_TRAIN++++++++++++++++++++++++++++++++++++++++++++++++")



                # box_preds = ret_dict["box_preds"]
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"]
                cls_neg_loss = ret_dict["cls_neg_loss"]
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                dir_loss_reduced = ret_dict["dir_loss_reduced"]
                cared = ret_dict["cared"]
                labels = example_paddle["labels"]
                if train_cfg.ENABLE_MIXED_PRECISION:
                    loss *= loss_scale
                
                # print("++++++++++++++++++++++++++++++++++++START LOSS_BACKWARD++++++++++++++++++++++++++++++++++++++++++++++++")
                loss.backward()
                # paddle.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                # print("++++++++++++++++++++++++++++++++++++STAR UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                mixed_optimizer.step()
                mixed_optimizer.clear_grad()
                # print("++++++++++++++++++++++++++++++++++++OVER UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                net.update_global_step()
                # print("++++++++++++++++++++++++++++++++++++STAR UPDATE_METRICS++++++++++++++++++++++++++++++++++++++++++++++++")
                net_metrics = net.update_metrics(
                    cls_loss_reduced,
                    loc_loss_reduced, 
                    cls_preds,
                    labels, cared)
                # print("++++++++++++++++++++++++++++++++++++OVER UPDATE_METRICS++++++++++++++++++++++++++++++++++++++++++++++++")
                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0).astype("float32")[0].sum().cpu().numpy())
                num_neg = int((labels == 0).astype("float32")[0].sum().cpu().numpy())
                if 'anchors_mask' not in example_paddle:
                    num_anchors = example_paddle['anchors'].shape[1]
                else:
                    num_anchors = int(example_paddle['anchors_mask'].astype("float32")[0].sum())
                # print("++++++++++++++++++++++++++++++++++++START GET_GLOBAL_STEP++++++++++++++++++++++++++++++++++++++++++++++++")
                global_step = net.get_global_step()
                # print("++++++++++++++++++++++++++++++++++++OVER GET_GLOBAL_STEP++++++++++++++++++++++++++++++++++++++++++++++++")
                if global_step % display_step == 0:
                    loc_loss_elem = [
                        float(loc_loss[:,:,i].sum().detach().cpu().numpy() /
                        batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["step"] = global_step
                    metrics["steptime"] = step_time
                    # print("++++++++++++++++++++++++++++++++++++START METRICS_UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                    metrics.update(net_metrics)
                    # print("++++++++++++++++++++++++++++++++++++OVER METRICS_UPDATE++++++++++++++++++++++++++++++++++++++++++++++++")
                    metrics["loss"] = {}
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(cls_neg_loss.detach().cpu().numpy())

                    if model_cfg.BACKBONE.use_direction_classifier:
                        metrics["loss"]["dir_rt"] = float(dir_loss_reduced.detach().cpu().numpy())
                    metrics["num_vox"] = int(example_paddle["voxels"].shape[0])
                    metrics["num_pos"] = int(num_pos)
                    metrics["num_neg"] = int(num_neg)
                    metrics["num_anchors"] = int(num_anchors)
                    metrics["lr"] = float(mixed_optimizer.get_lr())

                    metrics["image_idx"] = example['image_idx'][0]
                    flatted_metrics = flat_nested_json_dict(metrics)
                    flatted_summarys = flat_nested_json_dict(metrics, "/")
                    metrics_str_list = []
                    for k,v in flatted_metrics.items():
                        if isinstance(v,float):
                            metrics_str_list.append(f"{k}={v:.3}")
                        elif isinstance(v, (list, tuple)):
                            if v and isinstance(v[0], float):
                                v_str = ', '.join([f"{e:.3}" for e in v])
                                metrics_str_list.append(f"{k}=[{v_str}]")
                            else:
                                metrics_str_list.append(f"{k}={v}")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    log_str = ', '.join(metrics_str_list)
                    # print("++++++++++++++++++++++++++++++++++++STAR EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
                    # print("++++++++++++++++++++++++++++++++++++STAR EVAL++++++++++++++++++++++++++++++++++++++++++++++++", file=logf)
                    print(log_str)
                    print(log_str, file=logf)
                pd_elasped_time = time.time() - pd_start_time
                if pd_elasped_time > train_cfg.SAVE_CHECKPOINTS_SECS:
                    paddle.save(net.state_dict(), os.path.join(model_dir, "pointpillars.pdparams"))
                    paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "pointpillars.pdopt"))
                    pd_start_time = time.time()

            total_step_elapsed += steps
            # print("++++++++++++++++++++++++++++++++++++START SAVE++++++++++++++++++++++++++++++++++++++++++++++++")
            paddle.save(net.state_dict(), os.path.join(model_dir, "pointpillars.pdparams"))
            paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "pointpillars.pdopt"))
            # Ensure that all evaluation points are saved forever
            paddle.save(net.state_dict(), os.path.join(eval_checkpoint_dir, "pointpillars.pdparams"))
            paddle.save(mixed_optimizer.state_dict(), os.path.join(eval_checkpoint_dir, "pointpillars.pdopt"))
            # print("++++++++++++++++++++++++++++++++++++OVER SAVE++++++++++++++++++++++++++++++++++++++++++++++++")

            # net.eval()
            # result_path_step = result_path / f"step_{net.get_global_step()}"
            # result_path_step.mkdir(parents=True, exist_ok=True)
            # print("++++++++++++++++++++++++++++++++++++OVER EVAL++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("++++++++++++++++++++++++++++++++++++OVER EVAL++++++++++++++++++++++++++++++++++++++++++++++++", file=logf)
            # t = time.time()
            # dt_annos = []
            # print("**********************************************************************************")
            # prog_bar = ProgressBar()
            # print("**********************************************************************************")
            # prog_bar.start(len(eval_dataset) // eval_input_cfg.BATCH_SIZE + 1)
            # print("**********************************************************************************")
            # for example in iter(eval_dataloader):
            #     example = example_convert_to_paddle(example, float_dtype)
            #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #     if pickle_result:
            #             dt_annos += predict_kitti_to_anno(
            #                 net, example, class_names, center_limit_range,
            #                 model_cfg.LIDAR_INPUT)
            #     else:
            #         print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            #         _predict_kitti_to_file(net, example, result_path_step,
            #                                     class_names, center_limit_range,
            #                                     model_cfg.LIDAR_INPUT)
            #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #     prog_bar.print_bar()
            #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # sec_per_ex = len(eval_dataset) / (time.time() - t)
            # print("**********************************************************************************")
            # print(f"avg forward time per example: {net.avg_forward_time:.3f}")
            # print(f"avg postprocess time per example: {net.avg_postprocess_time:.3f}")
            # print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
            # print(f'generate label finished({sec_per_ex:.2f}/s). start eval:', file=logf)
            # gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
            # if not pickle_result:
            #     dt_annos = kitti.get_label_annos(result_path_step)
            #     result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(gt_annos, dt_annos, class_names,
            #                                                                         return_data=True)
            # print(result, file=logf)
            # print(result)
            
            # result = get_coco_eval_result(gt_annos, dt_annos, class_names)
            # print(result, file=logf)
            # print(result)
            # net.train()
        

    except Exception as e:
        paddle.save(net.state_dict(), os.path.join(model_dir, "pointpillars.pdparams"))
        paddle.save(mixed_optimizer.state_dict(), os.path.join(model_dir, "pointpillars.pdopt"))


def example_convert_to_paddle(example, dtype=paddle.float32) -> dict:
    example_paddle = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_paddle[k] = paddle.to_tensor(v, dtype=dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.int32)
        elif k in ["anchors_mask"]:
            example_paddle[k] = paddle.to_tensor(
                v, dtype=paddle.bool)
        else:
            example_paddle[k] = v

    return example_paddle

def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        if preds_dict["bbox"] is not None:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            box_2d_preds = preds_dict["bbox"].cpu().numpy()
            box_preds = preds_dict["box3d_camera"].cpu().numpy()
            scores = preds_dict["scores"].cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].cpu().numpy()
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].cpu().numpy()
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos


if __name__ == '__main__':
    fire.Fire()

# python train.py train --cfg_file=/home/hova/Lidardet/params/configs/pointpillars_kitti_car_xy16.yaml --model_dir=./logs/pp_0827